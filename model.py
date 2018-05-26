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
            self._make_upsample(256, 64),
            self._make_upsample(64, 16),
            nn.Conv2d(16, 16, (1, 1)),
            nn.Sigmoid()
        )

        del res50.avgpool
        del res50.fc
        del res50

    def _make_upsample(self, in_c, out_c):
        up = nn.Upsample(scale_factor=2, mode='bilinear')
        conv = nn.Conv2d(in_c, out_c, (3, 3), padding=1)
        bn = nn.BatchNorm2d(out_c)
        act = nn.ReLU()
        # nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        return nn.Sequential(up, conv, bn, act)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PoseEstimator(object):
    def __init__(self, ckpt_dir, device):
        super().__init__()
        self.ckpt_dir = pathlib.Path(ckpt_dir)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.epoch_dir = None
        self.ep = -1

        self.device = device
        self.model = PoseModel().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.BCELoss()

        print(self.model)
        print('CKPT:', self.ckpt_dir)

    def _train(self):
        self.msg.update({
            'loss': util.RunningAverage(),
        })
        self.model.train()
        for img_batch, lbl_batch, tag_batch in iter(self.train_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)
            tag_batch = tag_batch.to(self.device)

            self.optimizer.zero_grad()
            out_batch = self.model(img_batch)
            loss = self.criterion(out_batch, lbl_batch)
            loss.backward()
            self.optimizer.step()

            self.msg['loss'].update(loss)
            self.pbar.update(len(img_batch))
            self.pbar.set_postfix(self.msg)

    def _valid(self):
        self.msg.update({
            'val_loss': util.RunningAverage(),
        })
        self.model.eval()
        for img_batch, lbl_batch, tag_batch in iter(self.valid_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)
            tag_batch = tag_batch.to(self.device)

            out_batch = self.model(img_batch)
            loss = self.criterion(out_batch, lbl_batch)

            self.msg['val_loss'].update(loss)
        self.pbar.set_postfix(self.msg)

    def _vis(self):
        idx = 0
        self.model.eval()
        for img_batch, lbl_batch, tag_batch in iter(self.vis_loader):
            out_batch = self.model(img_batch.to(self.device)).cpu()

            B = len(img_batch)
            for i in range(B):
                vis = torch.cat([lbl_batch[i], out_batch[i]])
                vis = vis.unsqueeze(1) # (32, 1, H, W)
                img_path = self.epoch_dir / f'{idx:05d}.img.jpg'
                vis_path = self.epoch_dir / f'{idx:05d}.vis.jpg'
                save_image(img_batch[i], str(img_path))
                save_image(vis, str(vis_path), pad_value=1.0)
                idx += 1

    def _log(self):
        new_row = dict((k, v.avg) for k, v in self.msg.items())
        self.log = self.log.append(new_row, ignore_index=True)
        self.log.to_csv(str(self.ckpt_dir / 'log.csv'))
        # plot loss
        fig, ax = plt.subplots(dpi=100)
        self.log[['loss', 'val_loss']].plot(ax=ax)
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
