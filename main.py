import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
import utils
from thop import profile

from mydatasets import VideoDataset
from model_block.transformer import Transformer
from model_block.MOE import MultiTaskModel
from eval import test_model, calculate_heart_rate, calculate_repiration_rate
import MyLoss
from tqdm import tqdm

import pynvml
import warnings

warnings.simplefilter('ignore')
torch.autograd.set_detect_anomaly(True)

args = utils.get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

batch_size = args.batchsize
learning_rate = args.lr
max_iter = args.max_iter

root_dir = r'/remote-home/hao.lu/jywang/Data/Multi_Fatigue_Cognitive_Processed_old/train'
num_frames_per_sample = args.num_frames_per_sample
stride = args.stride

dataset = VideoDataset(root_dir, num_frames_per_sample, stride)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_dir = r'/remote-home/hao.lu/jywang/Data/Multi_Fatigue_Cognitive_Processed_old/test'
test_dataset = VideoDataset(test_dir, num_frames_per_sample, stride)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

pynvml.nvmlInit()
flag = 0
max_g = []
spaces = []
GPU = '10'
for gpu in range(8):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_Gpu = meminfo.free / 1024 / 1024 / 1024
    if free_Gpu > 40:
        flag = 1
        GPU = str(gpu)
        print("GPU:", GPU)
        print("free_Gpu:", free_Gpu)
        max_g = GPU
        break
    print("GPU:", gpu)
    print("free_Gpu:", free_Gpu)

if args.GPU != 10 and GPU == '10':
    GPU = str(args.GPU)
if torch.cuda.is_available():
    device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')
    print('on GPU ', GPU)
else:
    print('on CPU')

model = MultiTaskModel(block_num=args.num_block, num_experts=args.num_expert)
model = model.to(device)

model.count_parameters()
criterion1 = MyLoss.TruncatedLoss(trainset_size=len(dataset)).to(device)
criterion2 = MyLoss.TruncatedLoss(trainset_size=len(dataset)).to(device)

criterion_hr = nn.SmoothL1Loss().to(device)
criterion_rr = nn.SmoothL1Loss().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
iter_0 = dataloader.__iter__()
iter_per_epoch_0 = len(iter_0)
print('Focal Loss Version')
with tqdm(range(max_iter + 1)) as it:
    for iter_num in it:
        model.train()
        if (iter_num % iter_per_epoch_0 == 0):
            iter_0 = dataloader.__iter__()
        (frames_left_eye, frames_right_eye, frames_mouth, facials, STMaps, labels, labels_subject,
         subject_idx), idx = iter_0.__next__()

        frames_left_eye = frames_left_eye.permute(0, 4, 1, 2, 3).float()
        frames_right_eye = frames_right_eye.permute(0, 4, 1, 2, 3).float()
        frames_mouth = frames_mouth.permute(0, 4, 1, 2, 3).float()
        facials = facials.float()
        STMaps = STMaps.permute(0, 3, 1, 2).float()
        frames_left_eye = frames_left_eye.to(device)
        frames_right_eye = frames_right_eye.to(device)
        frames_mouth = frames_mouth.to(device)
        facials = facials.to(device)
        STMaps = STMaps.to(device)
        subject_idx = subject_idx.long().to(device)

        drowsiness = labels_subject[:, 0].unsqueeze(1).long().to(device)
        cognitive = labels_subject[:, 1].unsqueeze(1).long().to(device)
        hr = labels_subject[:, 2].unsqueeze(1).float().to(device)
        rr = labels_subject[:, 3].unsqueeze(1).float().to(device)

        outputs, feats, dis = model(
            frames_left_eye,
            frames_right_eye,
            frames_mouth,
            STMaps, facials)
        output_drowsiness = outputs[0]
        output_cognitive = outputs[1]
        output_hr = outputs[2]
        output_rr = outputs[3]

        loss1 = criterion1(output_drowsiness, drowsiness,
                           idx)
        loss2 = criterion2(output_cognitive, cognitive,
                           idx)
        loss_hr = criterion_hr(output_hr, hr)
        loss_rr = criterion_rr(output_rr, rr)

        loss_prior = MyLoss.reg_loss(output_drowsiness, output_cognitive)

        k = 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0
        loss = loss1 + loss2 + loss_hr + loss_rr + 0.001 * k * loss_prior

        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            loss.backward()
        optimizer.step()

        if iter_num % 10 == 0:
            print(
                'Iteration:' + str(iter_num + 1) \
                + ' | Overall Loss:  ' + str(loss.data.cpu().numpy()) \
                + ' | Drowsiness:  ' + str(loss1.data.cpu().numpy()) \
                + ' | Cognitive:  ' + str(loss2.data.cpu().numpy()) \
                + ' | HR:  ' + str(loss_hr.data.cpu().numpy()) \
                + ' | RR:  ' + str(loss_rr.data.cpu().numpy()) \
                + ' | Prior:  ' + str(loss_prior.data.cpu().numpy())
            )

        if iter_num % 500 == 0 and iter_num > 0:
            print('Evaluation results at :{}'.format(iter_num))
            test_model(model, test_dataloader, device, criterion1, criterion2)
