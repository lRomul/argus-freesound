import torch.nn as nn

nonlinearity = nn.ReLU


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, num_classes, input_channels=3, base_size=32, dropout=0.25):
        super(FeatureExtractor, self).__init__()
        self.input_channels = input_channels
        self.base_size = base_size
        s = base_size
        self.dropout = dropout
        self.input_conv = BasicConv2d(input_channels, s, 1)
        self.conv_1 = BasicConv2d(s * 1, s * 1, 3, padding=1)
        self.conv_2 = BasicConv2d(s * 1, s * 1, 3, padding=1)
        self.conv_3 = BasicConv2d(s * 1, s * 2, 3, padding=1)
        self.conv_4 = BasicConv2d(s * 2, s * 2, 3, padding=1)
        self.conv_5 = BasicConv2d(s * 2, s * 4, 3, padding=1)
        self.conv_6 = BasicConv2d(s * 4, s * 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(p=dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(s * 4, num_classes)

    def forward(self, x):
        x = self.input_conv(x)

        x = self.conv_1(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.dropout2d(x)

        x = self.conv_3(x)
        x = self.pool(x)
        x = self.conv_4(x)
        x = self.pool(x)
        x = self.dropout2d(x)

        x = self.conv_5(x)
        x = self.pool(x)
        x = self.conv_6(x)
        x = self.pool(x)
        x = self.dropout2d(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
