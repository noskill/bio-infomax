import torch
import torch.nn as nn


class TinyClass(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        mid = 4096 * 2
        self.layer0 = nn.Linear(n_in, mid)
        self.bn0 = nn.BatchNorm1d(mid)
        self.layer1 = nn.Linear(mid, mid)
        self.bn1 = nn.BatchNorm1d(mid)
        self.layer2 = nn.Linear(mid, n_out)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.bn0(self.layer0(x)))
        x = self.relu(self.bn1(self.layer1(x)))
        return torch.nn.functional.log_softmax(self.layer2(x), dim=1)


class TinyConv(nn.Module):
    def __init__(self, conv_in, n_out):
        super().__init__()
        mid = 256
        self.conv1 = nn.Conv2d(conv_in, mid, 2, 1)
        self.conv2 = nn.Conv2d(conv_in, mid, 2, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()
        self.layer3 = nn.Linear(mid, mid * 10)
        self.layer4 = nn.Linear(mid * 10, n_out)

    def forward(self, M):
        x = self.activation(self.conv1(M))
        x = self.activation(self.conv2(x))
        x = self.pool(x).squeeze()
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        return torch.nn.functional.log_softmax(x, dim=1)

