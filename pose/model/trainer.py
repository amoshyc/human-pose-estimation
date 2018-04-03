import pathlib
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from . import ae

class RunningAvg(object):
    def __init__(self):
        self.sum = 0.0
        self.iter = 0

    def update(self, x):
        self.sum += x
        self.iter += 1
    
    def __str__(self):
        if self.iter == 0:
            return '-'
        return f'{self.sum / self.iter:.3f}'


class Trainer(object):
    def __init__(self):
        self.net = ae.Net()
        self.optimizer = optim.Adam(self.net.parameters())
        self.seg_criterion = ae.seg_criterion
        self.tag_criterion = ae.tag_criterion

        self.msg = None
        self.pbar = None

    def __train(self):
        self.net.train()
        self.msg.update({
            'loss': RunningAvg(),
            'seg_loss': RunningAvg(),
            'tag_loss': RunningAvg(),
        })
        for inp_batch, hms_batch, jts_batch in next(self.trainloader):
            inp_var = Variable(inp_batch.cuda(), requires_grad=True)
            hms_var = Variable(hms_batch.cuda(), requires_grad=False)

            out_var = self.net(inp_var)
            seg_loss = self.seg_criterion(out_var[:, :-1, ...], hms_var)
            tag_loss = self.tag_criterion(out_var[:, -1, ...], jts_batch)
            loss = seg_loss + tag_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.update()

            self.msg['loss'].update(loss.data[0])
            self.msg['seg_loss'].update(seg_loss.data[0])
            self.msg['tag_loss'].update(tag_loss.data[0])
            self.pbar.set_postfix(**self.msg)
            self.pbar.update(len(inp_batch))

    def __valid(self):
        pass

    def __vis(self):
        pass

    def fit(self, trainset=None, validset=None, visset=None, epochs=50):
        # TODO

        self.net = self.net.cuda()

        for ep in range(epochs):
            tqdm_args = {
                'total': len(trainset),
                'desc': f'Epoch {ep:03d}/{epochs:03d}',
                'ascii': True
            }
            with tqdm(**tqdm_args) as self.pbar:
                self.msg = dict()
                self.__train()
                # self.__valid()
                # self.__vis()

        self.net = self.net.cpu()