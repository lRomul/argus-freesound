import torch
from torch import nn
import torch.nn.functional as F


class SEScale(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        channel = in_channels
        self.fc1 = nn.Linear(channel, reduction)
        self.fc2 = nn.Linear(reduction, channel)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class FCNet(nn.Module):
    def __init__(self, in_channels, num_classes,
                 base_size=64, reduction_scale=8,
                 p_dropout=0.2):
        super().__init__()
        self.p_dropout = p_dropout

        self.scale = SEScale(in_channels, in_channels // reduction_scale)
        self.linear1 = nn.Linear(in_channels, base_size*2)
        self.relu1 = nn.PReLU()
        self.linear2 = nn.Linear(base_size*2, base_size)
        self.relu2 = nn.PReLU()
        self.fc = nn.Linear(base_size, num_classes)

    def forward(self, x):
        x = self.scale(x) * x

        x = self.linear1(x)
        x = self.relu1(x)
        if self.p_dropout is not None:
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.linear2(x)
        x = self.relu2(x)
        if self.p_dropout is not None:
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.fc(x)
        return x
