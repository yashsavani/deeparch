import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.profiler as profiler

from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader
import train_model
import argparse
import pickle as pkl
import os

device = torch.device("cuda:0")

def main(args):
    args.autocast = True if args.autocast.lower() == 'true' else False
    trainloader, _ = train_model.get_data(args.batch_size, 224 if args.model_type == 'resnet' else -1)
    model = train_model.get_model(args.num_layers, args.model_type).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        acc = (yhat.argmax(dim=1) == y).float().mean()
        return loss.item(), acc.item()

    try:
        for epoch in range(1):
            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device)
                if args.autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        loss, acc = train_step(x, y)
                else:
                    loss, acc = train_step(x, y)
                break
        mem = torch.cuda.max_memory_allocated(device)
    except:
        mem = None
    
    if os.path.exists("mem_stats.pkl"):
        with open('mem_stats.pkl', 'rb') as f:
            mem_stats = pkl.load(f)
    else:
        mem_stats = {}

    mem_stats[args.model_type, args.num_layers, args.batch_size, args.autocast] = mem
    with open('mem_stats.pkl', 'wb') as f:
        pkl.dump(mem_stats, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=50)
    parser.add_argument('--model-type', type=str, default='resnet')
    parser.add_argument('--autocast', type=str, default='false')

    args = parser.parse_args()

    main(args)
