import torch
import torch.nn
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


def train(model, data, device, optimizer, criterion):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def get_train_loader(batch_size=32):
    transform = T.Compose(
        [T.Resize(224),
         T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader

def get_mem(model, batch_size, device, optimizer, criterion, logdir):
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/{logdir}"),
        profile_memory=True,
        record_shapes=True,
        with_stack=True)
    prof.start()

    train_loader = get_train_loader(batch_size=batch_size)
    for step, batch_data in enumerate(train_loader):
        if step == 4:
            break
        loss = train(model, batch_data, device, optimizer, criterion)
        prof.step()
    prof.stop()
    return torch.cuda.max_memory_allocated(device)

def get_model(num_layers):
    if num_layers == 18:
        model = torchvision.models.resnet18(pretrained=True)
    elif num_layers == 34:
        model = torchvision.models.resnet34(pretrained=True)
    elif num_layers == 50:
        model = torchvision.models.resnet50(pretrained=True)
    elif num_layers == 101:
        model = torchvision.models.resnet101(pretrained=True)
    elif num_layers == 152:
        model = torchvision.models.resnet152(pretrained=True)
    return model

def update_mem_stats(layers, batch_size, mem):
    if os.path.exists("mem_stats.pkl"):
        with open("mem_stats.pkl", "rb") as f:
            mem_stats = pkl.load(f)
    else:
        mem_stats = {}

    if args.resnet_layers not in mem_stats:
        mem_stats[args.resnet_layers] = {}
    mem_stats[layers][batch_size] = mem

    with open("mem_stats.pkl", "wb") as f:
        pkl.dump(mem_stats, f)


parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

parser.add_argument('--batch-size', type=int)
parser.add_argument('--resnet-layers', type=int)

args = parser.parse_args()

device = torch.device("cuda:0")
model = get_model(args.resnet_layers).cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
mem = get_mem(model, args.batch_size, device, optimizer, criterion, logdir=f"resnet{args.resnet_layers}_batch{args.batch_size}")
update_mem_stats(args.resnet_layers, args.batch_size, mem)
