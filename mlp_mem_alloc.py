import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.profiler as profiler

from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(batch_size=64):
    train = CIFAR10(
        root='./data', train=True, download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    )
    trainloader = DataLoader(train, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    test = CIFAR10(
        root='./data', train=False, download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
            x = F.relu(fc(x))
        x = self.fcout(x)
        return x


def main():
    trainloader, testloader = get_data()
    model = MLP(3072, 1024, 10, 2).to(device)
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
            # activities=[
            #     profiler.ProfilerActivity.CPU,
            #     profiler.ProfilerActivity.CUDA,
            # ],
            schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
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
    main()
