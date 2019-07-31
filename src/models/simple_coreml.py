from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.avg_pool2d = nn.AvgPool2d(2)
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
        x = self.avg_pool2d(x)
        return x


class SimpleCoreML(nn.Module):
    def __init__(self, num_classes, base_size=64, dropout=0.2):
        super().__init__()
        self.base_size = 64

        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=base_size),
            ConvBlock(in_channels=base_size, out_channels=base_size*2),
            ConvBlock(in_channels=base_size*2, out_channels=base_size*4),
            ConvBlock(in_channels=base_size*4, out_channels=base_size*8),
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_size*8, base_size*2),
            nn.PReLU(),
            nn.BatchNorm1d(base_size*2),
            nn.Dropout(dropout/2),
            nn.Linear(base_size*2, num_classes),
        )
        self.avg_pool2d = nn.AvgPool2d((8, 16))

    def forward(self, x):
        # Input  size: [batch_size, 3, 128, 256]
        x = self.conv(x)
        x = self.avg_pool2d(x)
        x = x.view(x.size(0), self.base_size*8)
        x = self.fc(x)
        return x
