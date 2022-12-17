import torch
from mlp_mixer_pytorch import MLPMixer
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.models
import torchvision.transforms as T
import argparse
import numpy as np
import pickle as pkl
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models import MLP, ViT


def get_data(batch_size=32, resize=-1):
    transforms = [] if resize == -1 else [T.Resize(resize)]
    transforms += [
        T.ToTensor(),
        T.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2470, 0.2435, 0.2616])
    ]
    transform = T.Compose(transforms)
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader, testloader


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
    elif model_type.lower() == 'vit':
        model = ViT(768, depth=n_layers, heads=12)

    return model


def main():
    args.autocast = True if args.autocast.lower() == 'true' else False
    log_dir = f"runs/{args.model_type}-{args.n_layers}-batch={args.batch_size}-autocast={args.autocast}"
    writer = SummaryWriter(log_dir=log_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    device = torch.device("cuda:0")
    model = get_model(args.n_layers, args.model_type).cuda(device)
    trainloader, testloader = get_data(
        batch_size=args.batch_size, 
        resize=224 if args.model_type == 'resnet' else -1
    )
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        accu = (yhat.argmax(dim=1) == y).float().mean()
        return loss.item(), accu.item()

    def test_step(x, y):
        model.eval()
        with torch.no_grad():
            yhat = model(x)
            loss = criterion(yhat, y)
            acc = (yhat.argmax(dim=1) == y).float().mean()
        return loss.item(), acc.item()
    
    def test_loop():
        test_loss = 0.
        test_accu = 0.
        for i, (x, y) in enumerate(testloader):
            x = x.to(device)
            y = y.to(device)
            loss, accu = test_step(x, y)
            test_loss += loss
            test_accu += accu
        print(
            f"Epoch {epoch}: test_loss: {test_loss / len(testloader):.3f}, test_acc: {test_accu / len(testloader):.3f}")
        writer.add_scalar('Test/Loss', test_loss / len(testloader), n_iter)
        writer.add_scalar('Test/Accu', test_accu / len(testloader), n_iter)

    n_iter = 0
    for epoch in range(args.epochs):
        train_loss = 0.
        train_accu = 0.
        for i, (x, y) in enumerate(trainloader):
            if n_iter % 100 == 0:
                test_loop()
            x = x.to(device)
            y = y.to(device)
            if args.autocast:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss, accu = train_step(x, y)
            else:
                loss, accu = train_step(x, y)
            train_loss += loss
            train_accu += accu
            writer.add_scalar('Train/Loss', loss, n_iter)
            writer.add_scalar('Train/Accu', accu, n_iter)
            n_iter += 1
        print(
            f"Epoch {epoch}: train_loss: {train_loss / len(trainloader):.3f}, train_acc: {train_accu / len(trainloader):.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--device-number', type=str, default="0")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-layers', type=int, default=18)
    parser.add_argument('--model-type', type=str, default='resnet')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--autocast', type=str, default='False')
    

    args = parser.parse_args()
    main()

