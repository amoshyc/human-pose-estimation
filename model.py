import pathlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.utils import save_image

from tqdm import tqdm
from skimage import feature
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import util


class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        res50 = resnet50(pretrained=True)

        self.encoder = nn.Sequential(
            res50.conv1,
            res50.bn1,
            res50.relu,
            res50.maxpool,
            res50.layer1,
            res50.layer2,
            res50.layer3,
            res50.layer4,
        )

        self.decoder = nn.Sequential(
            self._make_upsample(2048, 1024),
            self._make_upsample(1024, 512),
            self._make_upsample(512, 256),
            nn.Conv2d(256, 17, (1, 1)),
            # nn.BatchNorm2d(16),
            nn.Sigmoid()
        )

        del res50.avgpool
        del res50.fc
        del res50

    def _make_upsample(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, (2, 2), stride=2),
            # nn.BatchNorm2d(out_c),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp_batch, lbl_batch):
        wgt_batch = torch.ones_like(lbl_batch, dtype=torch.float)
        wgt_batch[lbl_batch > 0] = 50.0
        dff_batch = (inp_batch - lbl_batch)**2
        return torch.mean(wgt_batch * dff_batch)


class TagLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, inp_batch, tag_batch):
        batch_size = inp_batch.size(0)
        losses = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        for i in range(batch_size):
            inp, tag = inp_batch[i], tag_batch[i]
            kpts = tag.nonzero()
            K, D = kpts.size(0), inp.size(0)
            rr, cc = kpts[:, 0], kpts[:, 1]
            inp_kpts = inp[:, rr, cc]                   # (D, K)
            tag_kpts = tag[rr, cc]                      # (K)

            # Ground Truth
            A = tag_kpts.expand(K, K)                   # (K, K)
            B = A.t()                                   # (K, K)
            tag_similarity = (A == B).float()           # (K, K)

            # Prediction
            A = inp_kpts.unsqueeze(1)                   # (D, 1, K)
            A = A.expand(D, K, K)                       # (D, K, K)
            B = inp_kpts.unsqueeze(2)                   # (D, K, 1)
            B = B.expand(D, K, K)                       # (D, K, K)
            diff = (A - B)**2.mean(dim=0)               # (K, K)
            inp_similarity = 2 / (1 + torch.exp(diff))  # (K, K)

            # weighted MSE loss
            losses[i] = WeightedMSELoss()(inp_similarity, tag_similarity)
        return losses.mean() # average over batch



class PoseEstimator(object):
    def __init__(self, ckpt_dir, device):
        super().__init__()
        self.ckpt_dir = pathlib.Path(ckpt_dir)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.epoch_dir = None
        self.ep = -1

        self.device = device
        self.model = PoseModel().to(self.device)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.lbl_criterion = WeightedMSELoss()
        self.tag_criterion = TagLoss(self.device)

        print(self.model)
        print('CKPT:', self.ckpt_dir)

    def _train(self):
        self.msg.update({
            'loss': util.RunningAverage(),
            'lbl_loss': util.RunningAverage(),
            'tag_loss': util.RunningAverage(),
        })
        self.model.train()
        for img_batch, lbl_batch, tag_batch in iter(self.train_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)
            tag_batch = tag_batch.to(self.device)

            self.optimizer.zero_grad()
            out_batch = self.model(img_batch)
            lbl_loss = self.lbl_criterion(out_batch[:, :16], lbl_batch)
            tag_loss = self.tag_criterion(out_batch[:, 16:], tag_batch)
            loss = lbl_loss + tag_loss
            loss.backward()
            self.optimizer.step()

            self.msg['loss'].update(loss)
            self.msg['lbl_loss'].update(lbl_loss)
            self.msg['tag_loss'].update(tag_loss)
            self.pbar.update(len(img_batch))
            self.pbar.set_postfix(self.msg)

    def _valid(self):
        self.msg.update({
            'val_loss': util.RunningAverage(),
            'val_lbl_loss': util.RunningAverage(),
            'val_tag_loss': util.RunningAverage(),
        })
        self.model.eval()
        for img_batch, lbl_batch, tag_batch in iter(self.valid_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)
            tag_batch = tag_batch.to(self.device)

            out_batch = self.model(img_batch)
            lbl_loss = self.lbl_criterion(out_batch[:, :16], lbl_batch)
            tag_loss = self.tag_criterion(out_batch[:, 16:], tag_batch)
            loss = lbl_loss + tag_loss

            self.msg['val_loss'].update(loss)
            self.msg['val_lbl_loss'].update(lbl_loss)
            self.msg['val_tag_loss'].update(tag_loss)
        self.pbar.set_postfix(self.msg)

    def _vis(self):
        idx = 0
        self.model.eval()
        for img_batch, lbl_batch, tag_batch in iter(self.vis_loader):
            out_batch = self.model(img_batch.to(self.device)).cpu()

            B = len(img_batch)
            for i in range(B):
                img_path = self.epoch_dir / f'{idx:05d}.img.jpg'
                save_image(img_batch[i], str(img_path))

                lbl = torch.cat([lbl_batch[i, :16], out_batch[i, :16]])
                lbl = lbl.unsqueeze(1) # (32, 1, H, W)
                lbl_path = self.epoch_dir / f'{idx:05d}.lbl.jpg'
                save_image(lbl, str(lbl_path), pad_value=1.0)

                tag = out_batch[i, 16:]
                tag_path = self.epoch_dir / f'{idx:05d}.tag.jpg'
                save_image(tag, str(tag_path))

                idx += 1

    def _log(self):
        new_row = dict((k, v.avg) for k, v in self.msg.items())
        self.log = self.log.append(new_row, ignore_index=True)
        self.log.to_csv(str(self.ckpt_dir / 'log.csv'))
        # plot loss
        fig, ax = plt.subplots(dpi=100)
        self.log.plot(ax=ax)
        fig.tight_layout()
        fig.savefig(str(self.ckpt_dir / 'loss.jpg'))
        # Close plot to prevent RE
        plt.close()
        # model
        torch.save(self.model, str(self.epoch_dir / 'model.pth'))

    def fit(self, train_dataset, valid_dataset, vis_dataset, epoch=200):
        self.train_loader = DataLoader(train_dataset,
                batch_size=24, shuffle=True, num_workers=1)
        self.valid_loader = DataLoader(valid_dataset,
                batch_size=10, shuffle=False, num_workers=1)
        self.vis_loader = DataLoader(vis_dataset,
                batch_size=10, shuffle=False, num_workers=1)

        self.log = pd.DataFrame()
        for self.ep in range(epoch):
            self.epoch_dir = (self.ckpt_dir / f'{self.ep:03d}')
            self.epoch_dir.mkdir()
            self.msg = dict()

            tqdm_args = {
                'total': len(train_dataset),
                'desc': f'Epoch {self.ep:03d}',
                'ascii': True,
            }
            with tqdm(**tqdm_args) as self.pbar:
                self._train()
                with torch.no_grad():
                    self._valid()
                    self._vis()
                    self._log()
