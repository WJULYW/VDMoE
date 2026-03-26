import torch
import torch.nn as nn
import torch.nn.functional as F
from model_block.MOE import *
import torch.optim as optim

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding),
                               dilation=(1, dilation))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding),
                               dilation=(1, dilation))
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     padding=(kernel_size - 1) * dilation_size, dilation=dilation_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Parts_FeatureEmbedding2D(nn.Module):
    def __init__(self, input_shape, output_dim=128):
        super(Parts_FeatureEmbedding2D, self).__init__()
        num_channels = [16, 32]
        self.tcn = TCN(input_shape[1], num_channels, kernel_size=3)
        height = input_shape[3] // 4
        width = input_shape[4] // 4
        self.fc = nn.Linear(32 * height * width, output_dim)

    def forward(self, x):
        x = x.reshape(-1, x.size(1), x.size(3), x.size(4))
        x = self.tcn(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(-1, 300, x.size(1))
        return x

class STMapFeatureEmbedding(nn.Module):
    def __init__(self):
        super(STMapFeatureEmbedding, self).__init__()
        num_channels = [64, 128]
        self.tcn = TCN(3, num_channels, kernel_size=3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 300))
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.tcn(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(batch_size, 300, -1)
        x = self.fc(x)
        return x

class KeypointTo128(nn.Module):
    def __init__(self):
        super(KeypointTo128, self).__init__()
        num_channels = [32]
        self.tcn = TCN(1, num_channels, kernel_size=2)
        self.fc1 = nn.Linear(32 * 106, 128)

    def forward(self, x):
        batch_size, frames, keypoints, coords = x.shape
        x = x.view(batch_size * frames, 1, keypoints, coords)
        x = self.tcn(x)
        x = x.view(batch_size * frames, -1)
        x = self.fc1(x)
        x = x.view(batch_size, frames, 128)
        return x

class FeatureEmbedding2D(nn.Module):
    def __init__(self):
        super(FeatureEmbedding2D, self).__init__()
        self.embedding1 = Parts_FeatureEmbedding2D(input_shape=(4, 3, 300, 25, 25))
        self.embedding2 = Parts_FeatureEmbedding2D(input_shape=(4, 3, 300, 25, 25))
        self.embedding3 = Parts_FeatureEmbedding2D(input_shape=(4, 3, 300, 15, 35))
        self.stmap_embedding = STMapFeatureEmbedding()
        self.facial_embedding = KeypointTo128()

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)
        x3 = self.embedding3(x3)
        x4 = self.stmap_embedding(x4)
        x5 = self.facial_embedding(x5)
        combined = torch.cat((x1, x2, x3, x4, x5), dim=2)
        return combined

model = MultiTaskModel().to('cuda:1')
x1 = torch.randn(4, 3, 300, 25, 25).to('cuda:1')
x2 = torch.randn(4, 3, 300, 25, 25).to('cuda:1')
x3 = torch.randn(4, 3, 300, 15, 35).to('cuda:1')
x4 = torch.randn(4, 3, 25, 300).to('cuda:1')
x5 = torch.randn(4, 300, 106, 2).to('cuda:1')

criterion1 = nn.CrossEntropyLoss().to('cuda:1')

drowsiness = torch.randn(4,).long().to('cuda:1')

output_drowsiness, output_cognitive, output_resp, output_hr, output_rr = model(x1, x2, x3, x4, x5)
loss = criterion1(output_drowsiness, drowsiness)

optimizer = optim.Adam(model.parameters(), lr=0.1)

optimizer.zero_grad()
with torch.autograd.detect_anomaly():
    loss.backward()
optimizer.step()
