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

