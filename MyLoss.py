import os
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import argparse
from torch.autograd import Variable
from numpy import random
import math
from scipy.signal import find_peaks, welch
from scipy import signal
from scipy.fft import fft

class P_loss3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):

        mean_preds = torch.mean(preds, dim=1, keepdim=True)
        mean_targets = torch.mean(targets, dim=1, keepdim=True)
        preds_diff = preds - mean_preds
        targets_diff = targets - mean_targets

        numerator = torch.sum(preds_diff * targets_diff, dim=1)

        preds_diff_square_sum = torch.sum(preds_diff ** 2, dim=1)
        targets_diff_square_sum = torch.sum(targets_diff ** 2, dim=1)
        denominator = torch.sqrt(preds_diff_square_sum * targets_diff_square_sum)
        pearson_correlation = numerator / denominator

        loss = 1 - pearson_correlation
        return loss.mean()

class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, ), requires_grad=False)

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        if len(targets.shape)<2:
            Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        else:
            Yg = torch.gather(p, 1, targets)

        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q) * \
               self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)

class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])

class GHMC_Loss(GHM_Loss):

    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target

class MultiFocalLoss(nn.Module):

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()
            prob = prob.view(-1, prob.size(-1))

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth
        logpt = torch.log(prob)
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss

class loss_CM(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, av, ratio=0.1):

        s0 = torch.linalg.svdvals(av[0])
        s0 = torch.div(s0, torch.sum(s0))
        cov_loss0 = torch.sum(s0[s0 < (ratio/128)])

        s1 = torch.linalg.svdvals(av[1])
        s1 = torch.div(s1, torch.sum(s1))
        cov_loss1 = torch.sum(s1[s1 < (ratio/128)])

        s2 = torch.linalg.svdvals(av[2])
        s2 = torch.div(s2, torch.sum(s2))
        cov_loss2 = torch.sum(s2[s2 < (ratio / 128)])

        s3 = torch.linalg.svdvals(av[3])
        s3 = torch.div(s3, torch.sum(s3))
        cov_loss3 = torch.sum(s3[s3 < (ratio / 128)])

        return (cov_loss0 + cov_loss1 + cov_loss2 + cov_loss3)/4

class NEST_TA(nn.Module):
    def __init__(self, device, Num_ref=4, std=5):
        super().__init__()
        self.Num_ref = Num_ref
        self.Std = std

    def cos_sim(self, l1, l2):
        l1_mod = torch.div(l1, 1e-10 + torch.norm(l1, p=2, dim=1).unsqueeze(1))
        l2_mod = torch.div(l2, 1e-10 + torch.norm(l2, p=2, dim=1).unsqueeze(1))
        sim = torch.mean(torch.sum(torch.mul(l1_mod, l2_mod), dim=1))
        return sim

    def Gaussian_Smooth(self, sample, mean, std=5):
        gmm_wight = torch.exp(-torch.abs(sample - mean) ** 2 / (2 * std ** 2)) / (torch.sqrt(torch.tensor(2 * math.pi)) * std)
        gmm_wight = gmm_wight / torch.sum(gmm_wight, dim=1).unsqueeze(1)
        return gmm_wight

    def forward(self, Struct, Label):
        batch_size = Struct.shape[0]
        Ref_Index = np.tile(np.arange(0, batch_size, 1), batch_size)

        Label_ref = Label[Ref_Index].reshape(batch_size, batch_size).detach()
        Res_abs = torch.abs(Label_ref - Label.unsqueeze(1))
        _, sort_index = torch.sort(Res_abs, dim=1, descending=False)

        Struct_ref = Struct[sort_index[:, 0:self.Num_ref]]
        Label_ref = Label[sort_index[:, 0:self.Num_ref]].detach()

        gmm_wight = self.Gaussian_Smooth(Label_ref, mean=Label.unsqueeze(1), std=self.Std)

        Struct_mean = torch.sum(torch.mul(gmm_wight.unsqueeze(-1), Struct_ref), dim=1)
        Label_mean = torch.sum(torch.mul(gmm_wight, Label_ref), dim=1)

        Label_res = Label_ref - Label_mean.unsqueeze(1)
        Struct_res = Struct_ref - Struct_mean.unsqueeze(1)

        Label_d = Label.unsqueeze(1) - Label_mean.unsqueeze(1)

        ratio = (gmm_wight * torch.div(Label_d, 1e-10 + Label_res)).unsqueeze(2)

        Struct_smooth = torch.sum(ratio * Struct_res, dim=1) + Struct_mean

        sim = self.cos_sim(Struct, Struct_mean)

        return 1 - sim

def reg_loss(P_fatigue, P_cognitive):
    P_fatigue = F.softmax(P_fatigue / 2, dim=1)
    P_cognitive = F.softmax(P_cognitive / 2, dim=1)

    loss = F.kl_div(P_fatigue.log(), P_cognitive, reduction='batchmean')

    return -1*loss
