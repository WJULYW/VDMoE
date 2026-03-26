import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        return out

class ResnetExpert(nn.Module):
    def __init__(self, layers=[2, 2, 2, 2], d_model=640):
        super(ResnetExpert, self).__init__()
        self.in_channels = 300
        self.layer1 = self._make_layer(BasicBlock1D, 300, layers[0])
        self.layer2 = self._make_layer(BasicBlock1D, 300, layers[1])
        self.layer3 = self._make_layer(BasicBlock1D, 300, layers[2])
        self.layer4 = self._make_layer(BasicBlock1D, 300, layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, d_model))

    def _make_layer(self, block, out_channels, blocks):

        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x.view(x.size(0), -1)
