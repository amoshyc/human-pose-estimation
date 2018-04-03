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
        z2 = self.d2(F.max_pooling2d(z1))
        z3 = self.d3(F.max_pooling2d(z2))
        z4 = self.d4(F.max_pooling2d(z3))
        z = F.max_pooling2d(z4)

        z1 = self.m1(z1)
        z2 = self.m2(z2)
        z3 = self.m3(z3)
        z4 = self.m4(z4)
        z = self.m5(z)

        z4 = self.u4(z4 + F.upsample(z, scale_factor=2, mode='bilinear'))
        z3 = self.u3(z3 + F.upsample(z4, scale_factor=2, mode='bilinear'))
        z2 = self.u2(z2 + F.upsample(z3, scale_factor=2, mode='bilinear'))
        z1 = self.u1(z1 + F.upsample(z2, scale_factor=2, mode='bilinear'))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = _make_block(3, 32, k=7, p=0, s=2)
        self.hg = Hourglass(32)

    def forward(self, x):
        x = self.pre(x)
        x = self.hg(x)
        return x


if __name__ == '__main__':
    net = Net().cuda()

    inp = torch.rand((10, 3, 256, 256))
    inp_var = Variable(inp.cuda(), requires_grad=True)
    out_var = Net(inp_var)
    
    print(out_var.size)