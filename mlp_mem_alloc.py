import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.profiler as profiler

from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader

import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(batch_size=64):
    train = CIFAR10(
        root='./data', train=True, download=True,
        transform=T.Compose([
            T.RandomResizedCrop(32, scale=(args.scale, 1.), ratio=(1., 1.)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
            T.ColorJitter(args.jitter, args.jitter, args.jitter),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            T.RandomErasing(p=args.reprob),
        ])
    )
    trainloader = DataLoader(train, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    test = CIFAR10(
        root='./data', train=False, download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
    )
    testloader = DataLoader(test, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    return trainloader, testloader


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_hid):
        super(MLP, self).__init__()
        self.fcin = nn.Linear(input_size, hidden_size)
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(num_hid)
        ])
        self.fcout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fcin(x)
        for i, fc in enumerate(self.fcs):
            x = F.relu(fc(x)+x)
        x = self.fcout(x)
        return x


def main(args):
    trainloader, testloader = get_data(args.batch_size)
    model = MLP(3072, args.hidden_size, 10, args.num_layers).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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

    def test_step(x, y):
        model.eval()
        with torch.no_grad():
            yhat = model(x)
            loss = criterion(yhat, y)
            acc = (yhat.argmax(dim=1) == y).float().mean()
        return loss.item(), acc.item()

    for epoch in range(1):
        train_loss = 0.
        train_acc = 0.
        prof = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=profiler.tensorboard_trace_handler(
                './logs/mlp_'
                f'hid-{args.num_layers}'
                f'_bs-{args.batch_size}'
                f'_hs-{args.hidden_size}'),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )
        prof.start()
        for i, (x, y) in enumerate(trainloader):
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            loss, acc = train_step(x, y)
            prof.step()
            train_loss += loss
            train_acc += acc
        prof.stop()
        print(
            f"Epoch {epoch}: train_loss: {train_loss / len(trainloader):.3f}, train_acc: {train_acc / len(trainloader):.3f}")

        test_loss = 0.
        test_acc = 0.
        for i, (x, y) in enumerate(testloader):
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            loss, acc = test_step(x, y)
            test_loss += loss
            test_acc += acc
        print(
            f"Epoch {epoch}: test_loss: {test_loss / len(testloader):.3f}, test_acc: {test_acc / len(testloader):.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=1024)

    parser.add_argument('--scale', type=float, default=0.75)
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--ra-m', type=int, default=8)
    parser.add_argument('--ra-n', type=int, default=1)
    parser.add_argument('--jitter', type=float, default=0.1)
    args = parser.parse_args()

    main(args)