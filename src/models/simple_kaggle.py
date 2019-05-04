import torch
from torch import nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se_module=False, conv_bias=True):
        super().__init__()
        self.se_module = se_module
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if self.se_module:
            self.se = SEModule(out_channels, 16)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        if self.se_module:
            x = self.se(x)
        return x


class SimpleKaggle(nn.Module):
    def __init__(self, num_classes, base_size=64, dropout=0.2,
                 se_module=False, conv_bias=True):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=base_size,
                      se_module=se_module, conv_bias=conv_bias),
            ConvBlock(in_channels=base_size, out_channels=base_size*2,
                      se_module=se_module, conv_bias=conv_bias),
            ConvBlock(in_channels=base_size*2, out_channels=base_size*4,
                      se_module=se_module, conv_bias=conv_bias),
            ConvBlock(in_channels=base_size*4, out_channels=base_size*8,
                      se_module=se_module, conv_bias=conv_bias),
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_size*8, base_size*2),
            nn.PReLU(),
            nn.BatchNorm1d(base_size*2),
            nn.Dropout(dropout/2),
            nn.Linear(base_size*2, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x
