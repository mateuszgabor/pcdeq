import torch.nn as nn
from blocks import (
    LinearPcDEQ1Block,
    LinearPcDEQ2Block,
    ConvPcDEQ1Block,
    ConvPcDEQ2Block,
)


class LinearPcDEQ1(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, ch)
        self.bn1 = nn.BatchNorm1d(ch)
        self.block = LinearPcDEQ1Block(ch, act)
        self.bn2 = nn.BatchNorm1d(ch)
        self.classifier = nn.Linear(ch, 10)
        self.blocks = [self.block]

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.block(x)
        x = self.bn2(x)
        x = self.classifier(x)
        return x

    def clamp(self):
        for block in self.blocks:
            block.clamp()


class LinearPcDEQ2(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, ch)
        self.bn1 = nn.BatchNorm1d(ch)
        self.block = LinearPcDEQ2Block(ch, act)
        self.bn2 = nn.BatchNorm1d(ch)
        self.classifier = nn.Linear(ch, 10)
        self.blocks = [self.block]

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.block(x)
        x = self.bn2(x)
        x = self.classifier(x)
        return x

    def clamp(self):
        for block in self.blocks:
            block.clamp()


class SingleConvPcDEQ1(nn.Module):
    def __init__(self, in_dim, in_ch, ch1, act, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ch1, 3, padding=1)
        self.block = ConvPcDEQ1Block(ch1, act, **kwargs)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm2d(ch1)
        self.bn2 = nn.BatchNorm2d(ch1)
        self.avg_pool = nn.AvgPool2d(8)
        self.out_dim = ch1 * (in_dim // 8) ** 2
        self.classifier = nn.Linear(self.out_dim, 10)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.blocks = [self.block]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.block(x)
        x = self.max_pool(x)
        x = self.bn2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def clamp(self):
        for block in self.blocks:
            block.clamp()


class SingleConvPcDEQ2(nn.Module):
    def __init__(self, in_dim, in_ch, ch1, act, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ch1, 3, padding=1)
        self.block = ConvPcDEQ2Block(ch1, act, **kwargs)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm2d(ch1)
        self.bn2 = nn.BatchNorm2d(ch1)
        self.avg_pool = nn.AvgPool2d(8)
        self.out_dim = ch1 * (in_dim // 8) ** 2
        self.classifier = nn.Linear(self.out_dim, 10)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.blocks = [self.block]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.block(x)
        x = self.max_pool(x)
        x = self.bn2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def clamp(self):
        for block in self.blocks:
            block.clamp()


class MultiConvPcDEQ1(nn.Module):
    def __init__(self, in_ch, ch1, ch2, ch3, act, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ch1, 3, padding=1)
        self.block1 = ConvPcDEQ1Block(ch1, act, **kwargs)
        self.downsample1 = nn.Conv2d(ch1, ch2, 3, stride=2)
        self.block2 = ConvPcDEQ1Block(ch2, act, **kwargs)
        self.downsample2 = nn.Conv2d(ch2, ch3, 3, stride=2)
        self.block3 = ConvPcDEQ1Block(ch3, act, **kwargs)
        self.avg_pool = nn.AvgPool2d(4)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(ch3, 10)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.bn2 = nn.BatchNorm2d(ch1)
        self.bn3 = nn.BatchNorm2d(ch2)
        self.bn4 = nn.BatchNorm2d(ch2)
        self.bn5 = nn.BatchNorm2d(ch3)
        self.bn6 = nn.BatchNorm2d(ch3)
        self.blocks = [self.block1, self.block2, self.block3]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.block1(x)
        x = self.max_pool(x)
        x = self.bn2(x)
        x = self.downsample1(x)
        x = self.bn3(x)
        x = self.block2(x)
        x = self.max_pool(x)
        x = self.bn4(x)
        x = self.downsample2(x)
        x = self.bn5(x)
        x = self.block3(x)
        x = self.max_pool(x)
        x = self.bn6(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def clamp(self):
        for block in self.blocks:
            block.clamp()


class MultiConvPcDEQ2(nn.Module):
    def __init__(self, in_ch, ch1, ch2, ch3, act, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ch1, 3, padding=1)
        self.block1 = ConvPcDEQ2Block(ch1, act, **kwargs)
        self.downsample1 = nn.Conv2d(ch1, ch2, 3, stride=2)
        self.block2 = ConvPcDEQ2Block(ch2, act, **kwargs)
        self.downsample2 = nn.Conv2d(ch2, ch3, 3, stride=2)
        self.block3 = ConvPcDEQ2Block(ch3, act, **kwargs)
        self.avg_pool = nn.AvgPool2d(4)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(ch3, 10)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.bn2 = nn.BatchNorm2d(ch1)
        self.bn3 = nn.BatchNorm2d(ch2)
        self.bn4 = nn.BatchNorm2d(ch2)
        self.bn5 = nn.BatchNorm2d(ch3)
        self.bn6 = nn.BatchNorm2d(ch3)
        self.blocks = [self.block1, self.block2, self.block3]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.block1(x)
        x = self.max_pool(x)
        x = self.bn2(x)
        x = self.downsample1(x)
        x = self.bn3(x)
        x = self.block2(x)
        x = self.max_pool(x)
        x = self.bn4(x)
        x = self.downsample2(x)
        x = self.bn5(x)
        x = self.block3(x)
        x = self.max_pool(x)
        x = self.bn6(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def clamp(self):
        for block in self.blocks:
            block.clamp()
