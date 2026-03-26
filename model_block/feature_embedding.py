import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

args = utils.get_args()

class WTSM(nn.Module):
    def __init__(self, n_segment=3, fold_div=3):
        super(WTSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.reshape(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, -1, :fold] = x[:, 0, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.reshape(nt, c, h, w)

class Parts_FeatureEmbedding2D2(nn.Module):
    def __init__(self, input_shape=(4, 3, 300, 25, 25), output_dim=128):
        super(Parts_FeatureEmbedding2D2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[1], out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.wstm = WTSM()

        height = input_shape[3] // 4
        width = input_shape[4] // 4

        self.fc = nn.Linear(32 * height * width, output_dim)

    def forward(self, x):
        x = x.reshape(-1, x.size(1), x.size(3), x.size(4))
        x = self.wstm(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(-1, 300, x.size(1))
        return x

class Parts_FeatureEmbedding2D(nn.Module):
    def __init__(self, input_shape, output_dim=128):
        super(Parts_FeatureEmbedding2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[1], out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        height = input_shape[3] // 4
        width = input_shape[4] // 4

        self.fc = nn.Linear(32 * height * width, output_dim)

    def forward(self, x):
        x = x.reshape(-1, x.size(1), x.size(3), x.size(4))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(-1, 300, x.size(1))
        return x

class STMapFeatureEmbedding(nn.Module):
    def __init__(self):
        super(STMapFeatureEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128, affine=False)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 300))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(batch_size, 300, -1)
        return x

class KeypointTo128(nn.Module):
    def __init__(self):
        super(KeypointTo128, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 2), stride=1, padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(32 * 106, 128)

    def forward(self, x):
        batch_size, frames, keypoints, coords = x.shape
        x = x.view(batch_size * frames, 1, keypoints, coords)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(batch_size * frames, -1)
        x = self.fc1(x)
        x = self.relu(x)
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
