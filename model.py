import pathlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import *
from torchvision.utils import save_image

import numpy as np
from tqdm import tqdm
from skimage import feature
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from coco import COCOKeypoint


class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = densenet121(pretrained=True)

        self.backbone = densenet.features
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (2, 2), stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, (2, 2), stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 64, (2, 2), stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 20, (1, 1)),
            nn.BatchNorm2d(20),
        )

    def forward(self, x):
        feature = self.backbone(x)
        output = self.decode(feature)
        lbl = F.tanh(output[:, :17, ...])
        tag = F.tanh(output[:, 17:, ...])
        return lbl, tag


class LblLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_batch, true_batch):
        wgt = torch.ones_like(pred_batch)
        wgt[true_batch > 0] = 100
        dis = (pred_batch - true_batch)**2
        return (dis * wgt).mean()


class TagLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_batch, kpt_batch, vis_batch, tag_batch):
        batch_size, D, lblH, lblW = pred_batch.size()
        device = pred_batch.device
        losses = torch.zeros(batch_size, dtype=torch.float, device=device)
        unnorm_term = torch.tensor([lblW, lblH], dtype=torch.float, device=device)

        for i in range(batch_size):
            pred = pred_batch[i]                # (D, dstH, dstW)
            viss = vis_batch[i].to(device)      # (n_people * 17,)
            tags = tag_batch[i].to(device)      # (n_people * 17,)
            kpts = kpt_batch[i].to(device)      # (n_people * 17, 2)
            kpts = kpts[viss > 0] * unnorm_term
            kpts = torch.floor(kpts).long()     # Don't use round -> index out of range
            true_ebd = tags[viss > 0]
            pred_ebd = pred[:, kpts[:, 0], kpts[:, 1]]
            K = true_ebd.size(0)

            A = true_ebd.expand(K, K)           # (K, K)
            B = A.t()                           # (K, K)
            true_similarity = (A == B).float()  # (K, K)

            A = pred_ebd.unsqueeze(1)           # (D, 1, K)
            A = A.expand(D, K, K)               # (D, K, K)
            B = pred_ebd.unsqueeze(2)           # (D, K, 1)
            B = B.expand(D, K, K)               # (D, K, K)
            exponent = ((A - B)**2).mean(dim=0) # (K, K)
            pred_similarity = 2 / (1 + torch.exp(exponent))

            wgt = torch.zeros(K, K, dtype=torch.float, device=device)
            wgt[(true_similarity > 0) | (pred_similarity > 0)] = 10.0
            dis = (pred_similarity - true_similarity)**2
            losses[i] = (dis * wgt).mean()
        return torch.mean(losses)


class RunningAverage(object):
    def __init__(self):
        super().__init__()
        self.iter = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = (self.avg * self.iter + x) / (self.iter + 1)
        self.iter += 1

    def __str__(self):
        if self.iter == 0:
            return '-'
        return f'{self.avg:.4f}'


class PoseEstimator:
    def __init__(self, log_dir, device):
        self.device = device
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.model = PoseModel().to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.decay = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10)
        self.lbl_criterion = LblLoss()
        self.tag_criterion = TagLoss()

    def fit(self, train_set, valid_set, vis_set, epoch=100):
        self.train_loader = DataLoader(train_set, batch_size=16,
            shuffle=True, collate_fn=COCOKeypoint.collate_fn, num_workers=4)
        self.valid_loader = DataLoader(valid_set, batch_size=16,
            shuffle=False, collate_fn=COCOKeypoint.collate_fn, num_workers=4)
        self.vis_loader = DataLoader(vis_set, batch_size=16,
            shuffle=False, collate_fn=COCOKeypoint.collate_fn, num_workers=4)

        self.log = pd.DataFrame()
        for self.ep in range(epoch):
            self.epoch_dir = (self.log_dir / f'{self.ep:03d}')
            self.epoch_dir.mkdir()
            self.msg = dict()

            tqdm_args = {
                'total': len(train_set) + len(valid_set),
                'desc': f'Epoch {self.ep:03d}',
                'ascii': True,
            }
            with tqdm(**tqdm_args) as self.pbar:
                self.decay.step()
                self._train()
                with torch.no_grad():
                    self._valid()
                    self._vis()
                    self._log()

    def _train(self):
        self.msg.update({
            'loss': RunningAverage(),
            'lbl_loss': RunningAverage(),
            'tag_loss': RunningAverage()
        })
        self.model.train()
        for img_batch, lbl_batch, kpt_batch, \
            vis_batch, tag_batch, box_batch in iter(self.train_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)

            self.optim.zero_grad()
            pred_lbl, pred_tag = self.model(img_batch)
            lbl_loss = self.lbl_criterion(pred_lbl, lbl_batch)
            tag_loss = self.tag_criterion(pred_tag, kpt_batch, vis_batch, tag_batch) * 0.005
            loss = lbl_loss + tag_loss
            loss.backward()
            self.optim.step()

            self.msg['loss'].update(loss.item())
            self.msg['lbl_loss'].update(lbl_loss.item())
            self.msg['tag_loss'].update(tag_loss.item())
            self.pbar.set_postfix(self.msg)
            self.pbar.update(len(img_batch))

    def _valid(self):
        self.msg.update({
            'val_loss': RunningAverage(),
            'val_lbl_loss': RunningAverage(),
            'val_tag_loss': RunningAverage()
        })
        self.model.eval()
        for img_batch, lbl_batch, kpt_batch, \
            vis_batch, tag_batch, box_batch in iter(self.valid_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)

            pred_lbl, pred_tag = self.model(img_batch)
            lbl_loss = self.lbl_criterion(pred_lbl, lbl_batch)
            tag_loss = self.tag_criterion(pred_tag, kpt_batch, vis_batch, tag_batch) * 0.005
            loss = lbl_loss + tag_loss

            self.msg['val_loss'].update(loss.item())
            self.msg['val_lbl_loss'].update(lbl_loss.item())
            self.msg['val_tag_loss'].update(tag_loss.item())
            self.pbar.update(len(img_batch))
        self.pbar.set_postfix(self.msg)

    def _vis(self):
        self.model.eval()
        idx = 0
        for img_batch, lbl_batch, kpt_batch, \
            vis_batch, tag_batch, box_batch in iter(self.vis_loader):
            pred_lbl, pred_tag = self.model(img_batch.to(self.device))
            pred_lbl = pred_lbl.cpu()
            pred_tag = pred_tag.cpu()
            pred_lbl = F.sigmoid(pred_lbl)
            pred_tag = F.sigmoid(pred_tag)
            batch_size, _, H, W = img_batch.size()
            pred_tag = F.upsample(pred_tag, (H, W))

            for i in range(batch_size):
                img = img_batch[i]
                vis_lbl = torch.cat((lbl_batch[i], pred_lbl[i]), dim=0).unsqueeze(1)
                vis_tag = pred_tag[i] * 0.7 + 0.3 * img
                save_image(img, f'{self.epoch_dir}/{idx:05d}.jpg')
                save_image(vis_lbl, f'{self.epoch_dir}/{idx:05d}_lbl.jpg', nrow=17, pad_value=1)
                save_image(vis_tag, f'{self.epoch_dir}/{idx:05d}_tag.jpg')
                idx += 1

    def _log(self):
        new_row = dict((k, v.avg) for k, v in self.msg.items())
        self.log = self.log.append(new_row, ignore_index=True)
        self.log.to_csv(str(self.log_dir / 'log.csv'))
        # plot loss
        fig, ax = plt.subplots(1, 3, dpi=100)
        self.log[['loss', 'val_loss']].plot(ax=ax[0])
        self.log[['lbl_loss', 'val_lbl_loss']].plot(ax=ax[1])
        self.log[['tag_loss', 'val_tag_loss']].plot(ax=ax[2])
        fig.tight_layout()
        fig.savefig(str(self.log_dir / 'loss.jpg'))
        plt.close()  # Close plot to prevent RE
        # model
        torch.save(self.model, str(self.epoch_dir / 'model.pth'))


if __name__ == '__main__':
    img_dir = '/store/COCO/val2017/'
    anno_path = '/store/COCO/annotations/person_keypoints_val2017.json'
    ds = COCOKeypoint(img_dir, anno_path, img_size=(384, 384), lbl_size=(96, 96))
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=ds.collate_fn, num_workers=1)

    device = torch.device('cuda')
    model = PoseModel().to(device)
    model = model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    pbar = tqdm(total=len(ds), ascii=True)
    for img_batch, lbl_batch, kpt_batch, vis_batch, tag_batch, box_batch in dl:
        img_batch = img_batch.to(device)
        lbl_batch = lbl_batch.to(device)
        pred_lbl, pred_tag = model(img_batch)

        optim.zero_grad()
        lbl_loss = LblLoss()(pred_lbl, lbl_batch)
        tag_loss = TagLoss()(pred_tag, kpt_batch, vis_batch, tag_batch)
        loss = lbl_loss + tag_loss
        loss.backward()
        optim.step()

        pbar.update(len(img_batch))
        pbar.set_postfix({
            'loss': loss.item(),
            'lbl_loss': lbl_loss.item(),
            'tag_loss': tag_loss.item()
        })
    pbar.close()
