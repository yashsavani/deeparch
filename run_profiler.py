import torch
from mlp_mixer_pytorch import MLPMixer
import torch.nn as nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
import argparse
import numpy as np
import pickle as pkl
import os
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_hid):
        super(MLP, self).__init__()
        self.fcin = nn.Linear(input_size, hidden_size)
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(num_hid)
        ])
        self.fcout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fcin(x)
        for i, fc in enumerate(self.fcs):
            x = F.relu(fc(x))
        x = self.fcout(x)
        return x


def train(model, data, device, optimizer, criterion):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def get_train_loader_resnet(batch_size=32):
    transform = T.Compose(
        [T.Resize(224),
         T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader

def get_train_loader(batch_size=32):
    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader


def get_mem(model, train_loader, device, optimizer, criterion, logdir):
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./logs/{logdir}"),
        profile_memory=True,
        record_shapes=True,
        with_stack=True)
    prof.start()

    for step, batch_data in enumerate(train_loader):
        if step == 25:
            break
        loss = train(model, batch_data, device, optimizer, criterion)
        prof.step()
    prof.stop()
    return torch.cuda.max_memory_allocated(device)

def get_model(n_layers, model_type='resnet'):
    if model_type == 'resnet':
        if n_layers == 18:
            model = torchvision.models.resnet18(pretrained=True)
        elif n_layers == 34:
            model = torchvision.models.resnet34(pretrained=True)
        elif n_layers == 50:
            model = torchvision.models.resnet50(pretrained=True)
        elif n_layers == 101:
            model = torchvision.models.resnet101(pretrained=True)
        elif n_layers == 152:
            model = torchvision.models.resnet152(pretrained=True)
        else:
            raise ValueError(f"n_layers={n_layers} not supported for resnet")
    elif model_type == 'mixer':
        model = MLPMixer(
            image_size = 32,
            channels = 3,
            patch_size = 16,
            dim = 512,
            depth = n_layers,
            num_classes = 10
        )
    elif model_type == 'mlp':
        model = MLP(3072, 1024, 10, n_layers)

    return model

def update_mem_stats(layers, batch_size, mem):
    if os.path.exists("mem_stats.pkl"):
        with open("mem_stats.pkl", "rb") as f:
            mem_stats = pkl.load(f)
    else:
        mem_stats = {}

    if args.n_layers not in mem_stats:
        mem_stats[args.resnet_layers] = {}
    mem_stats[layers][batch_size] = mem

    with open("mem_stats.pkl", "wb") as f:
        pkl.dump(mem_stats, f)


def main():
    device = torch.device("cuda:0")
    model = get_model(args.n_layers, args.model_type).cuda(device)
    if args.model_type == 'resnet':
        train_loader = get_train_loader_resnet(batch_size=args.batch_size)
    else:
        train_loader = get_train_loader(batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    mem = get_mem(model, train_loader, device, optimizer, criterion, logdir=f"{args.model_type}{args.n_layers}_batch{args.batch_size}")
    # update_mem_stats(args.resnet_layers, args.batch_size, mem)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--n-layers', type=int)
    parser.add_argument('--model-type', type=str, default='resnet')

    args = parser.parse_args()
    main()

