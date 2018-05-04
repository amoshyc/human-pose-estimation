import pathlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._make_conv(3, 8, k=3, s=2, p=1, a='leaky'),
            self._make_conv(8, 32, k=3, s=2, p=1, a='leaky'),
            self._make_conv(32, 32, k=3, s=1, p=1, a='leaky'),
            nn.Upsample(scale_factor=2),
            self._make_conv(32, 24, k=3, s=1, p=1, a='leaky'),
            nn.Upsample(scale_factor=2),
            self._make_conv(24, 17, k=3, s=1, p=1, a='leaky'),
            self._make_conv(17, 17, k=1, s=1, p=0, a='sigmoid'),
        )

    def _make_conv(self, in_c, out_c, k=1, s=1, p=0, a=None):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)]
        layers.append(nn.BatchNorm2d(out_c))
        if a == 'relu':
            layers.append(nn.ReLU())
            nn.init.kaiming_normal_(layers[0].weight, nonlinearity='relu')
        elif a == 'leaky':
            layers.append(nn.LeakyReLU())
            nn.init.kaiming_normal_(
                layers[0].weight, nonlinearity='leaky_relu')
        elif a == 'sigmoid':
            layers.append(nn.Sigmoid())
            nn.init.xavier_normal_(layers[0].weight)
        elif a is None:
            nn.init.xavier_normal_(layers[0].weight)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class RunningAverage(object):
    def __init__(self):
        super().__init__()
        self.iter = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = (self.avg * self.iter + x.item()) / (self.iter + 1)
        self.iter += 1

    def __str__(self):
        if self.iter == 0:
            return 'x'
        return f'{self.avg:.3f}'


class TagLoss(object):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def __call__(self, inp_batch, tag_batch):
        batch_size = inp_batch.size(0)
        losses = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        for i in range(batch_size):
            inp, tag = inp_batch[i], tag_batch[i]
            inp, tag = inp[0], tag[0]
            kpts = tag.nonzero()
            K = kpts.size(0)
            rr, cc = kpts[:, 0], kpts[:, 1]
            inp_kpts = inp[rr, cc]
            tag_kpts = tag[rr, cc]

            A = tag_kpts.expand(K, K)
            B = A.t()
            tag_similarity = (A == B).float()

            A = inp_kpts.unsqueeze(1)
            A = A.expand(K, K)
            B = inp_kpts.unsqueeze(0)
            B = B.expand(K, K)
            diff = ((A - B)**2)
            inp_similarity = 2 / (1 + torch.exp(diff))

            tag_similarity = tag_similarity.unsqueeze(0)
            inp_similarity = inp_similarity.unsqueeze(0)

            losses[i] = F.mse_loss(inp_similarity, tag_similarity)
        return losses.mean()


class PoseEstimator(object):
    def __init__(self, ckpt_dir, device):
        super().__init__()
        self.ckpt_dir = pathlib.Path(ckpt_dir)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.epoch_dir = None
        self.pbar = None
        self.msg = None

        self.device = device
        self.model = PoseModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.lbl_criterion = nn.MSELoss()
        self.tag_criterion = TagLoss(self.device)

        print(self.model)
        print('CKPT:', self.ckpt_dir)

    def _train(self):
        self.msg.update({
            'loss': RunningAverage(),
            'lbl_loss': RunningAverage(),
            'tag_loss': RunningAverage(),
        })
        self.model.train()
        for img_batch, lbl_batch, tag_batch in iter(self.train_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)
            tag_batch = tag_batch.to(self.device)

            self.optimizer.zero_grad()
            out_batch = self.model(img_batch)
            lbl_loss = self.lbl_criterion(out_batch[:, :16, ...], lbl_batch)
            tag_loss = self.tag_criterion(out_batch[:, 16:, ...], tag_batch)
            loss = lbl_loss + tag_loss
            loss.backward()
            self.optimizer.step()

            self.msg['loss'].update(loss)
            self.msg['lbl_loss'].update(lbl_loss)
            self.msg['tag_loss'].update(tag_loss)
            self.pbar.update(len(img_batch))
            self.pbar.set_postfix(**self.msg)

    # def fit(self, train_dataset, valid_dataset, vis_dataset, epoch=50):
    def fit(self, train_dataset, epoch=50):
        self.train_loader = DataLoader(train_dataset,
                batch_size=32, shuffle=True, num_workers=3)
        # self.valid_loader = DataLoader(valid_dataset,
        #         batch_size=10, shuffle=False, num_workers=1)
        # self.vis_loader = DataLoader(vis_dataset,
        #         batch_size=10, shuffle=False, num_workers=1)

        # self.log = pd.DataFrame()
        for ep in range(epoch):
            self.epoch_dir = (self.ckpt_dir / f'{ep:03d}')
            self.epoch_dir.mkdir()
            self.msg = dict()

            tqdm_args = {
                'total': len(train_dataset),
                'ascii': True,
                'desc': f'Epoch: {ep:03d}',
            }
            with tqdm(**tqdm_args) as self.pbar:
                self._train()
                # with torch.no_grad():
                #     self._valid()
                #     self._vis()
                #     self._log()
