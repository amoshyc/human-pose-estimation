import pathlib
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _make_block(in_c, out_c, k=3, p=0, s=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=p),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )


class Hourglass(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.d1 = _make_block(c, c, k=3, p=1)
        self.d2 = _make_block(c, c, k=3, p=1)
        self.d3 = _make_block(c, c, k=3, p=1)
        self.d4 = _make_block(c, c, k=3, p=1)

        self.u1 = _make_block(c, c, k=3, p=1)
        self.u2 = _make_block(c, c, k=3, p=1)
        self.u3 = _make_block(c, c, k=3, p=1)
        self.u4 = _make_block(c, c, k=3, p=1)

        self.m1 = _make_block(c, c, k=3, p=1)
        self.m2 = _make_block(c, c, k=3, p=1)
        self.m3 = _make_block(c, c, k=3, p=1)
        self.m4 = _make_block(c, c, k=3, p=1)
        self.m5 = nn.Sequential(
            _make_block(c, c, k=3, p=1),
            _make_block(c, c, k=3, p=1),
            _make_block(c, c, k=3, p=1)
        )

    def forward(self, x):
        z1 = self.d1(x)
        z2 = self.d2(F.max_pool2d(z1, 2))
        z3 = self.d3(F.max_pool2d(z2, 2))
        z4 = self.d4(F.max_pool2d(z3, 2))
        z = F.max_pool2d(z4, 2)

        z1 = self.m1(z1)
        z2 = self.m2(z2)
        z3 = self.m3(z3)
        z4 = self.m4(z4)
        z = self.m5(z)

        z4 = self.u4(z4 + F.upsample(z, scale_factor=2, mode='bilinear'))
        z3 = self.u3(z3 + F.upsample(z4, scale_factor=2, mode='bilinear'))
        z2 = self.u2(z2 + F.upsample(z3, scale_factor=2, mode='bilinear'))
        z1 = self.u1(z1 + F.upsample(z2, scale_factor=2, mode='bilinear'))
        return z1


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = _make_block(3, 32, k=7, p=3, s=2)
        self.hg = Hourglass(32)
        self.post = _make_block(32, 17, k=1)

    def forward(self, x):
        x = self.pre(x)
        x = self.hg(x)
        x = self.post(x)
        x[:, :-1, ...] = F.sigmoid(x[:, :-1, ...])
        return x


def tag_criterion(pred_batch, kpt_batch):
    # TODO, currently incorrect
    loss = 0.0
    for pred, kpt in zip(pred_batch, kpt_batch):
        tags = pred[kpt > 0]
        re = torch.mean(tags)
        A = re.expand(len(re), len(re))
        B = A.t()
        loss1 = torch.mean((tags - re)**2)
        loss2 = torch.mean(torch.exp((-1/2) * (A - B)**2))
        loss += (loss1 + loss2)
    return loss


def seg_criterion(pred_batch, hmp_batch):
    return F.mse_loss(pred_batch, hmp_batch)


if __name__ == '__main__':
    net = Net().cuda()

    inp = torch.rand((10, 3, 256, 256))
    inp_var = Variable(inp.cuda(), requires_grad=True)
    out_var = net(inp_var)
    
    print(out_var.size())