from distutils.dep_util import newer_group
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class tripletLoss(nn.Module):
    def __init__(self, margin):
        super(tripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        # original setting
        distance_pos = (anchor - pos).pow(2).sum(1)
        distance_neg = (anchor - neg).pow(2).sum(1)
        # distance_pos = torch.var(anchor - pos) / torch.var(anchor)
        # distance_neg = torch.var(anchor - neg) / torch.var(anchor)
        # print(distance_neg)
        # print(distance_pos - distance_neg)
        loss = F.relu(distance_pos - distance_neg + self.margin)
        return loss.mean(), self.triplet_correct(distance_pos, distance_neg)

    def triplet_correct(self, d_pos, d_neg):
        return (d_pos < d_neg).sum()


class combLoss(nn.Module):
    def __init__(self, margin, l = 1):
        super(combLoss, self).__init__()
        self.margin = margin
        self.l = l

    def forward(self, anchor, pos, neg):
        # print("-----")
        # print((anchor - pos).pow(2).shape)
        distance_pos = (anchor - pos).pow(2).sum(1)
        # print(distance_pos.shape)
        distance_neg = (anchor - neg).pow(2).sum(1)
        distance_cen = (neg - anchor * 0.5 - pos * 0.5).pow(2).sum(1)
        loss = F.relu(distance_pos - self.l * distance_cen + self.margin)
        return loss.mean(), self.triplet_correct(distance_pos, distance_neg)

    def triplet_correct(self, d_pos, d_neg):
        return (d_pos < d_neg).sum()


