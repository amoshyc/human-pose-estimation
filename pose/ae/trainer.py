import pathlib
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from . import model

class RunningAvg(object):
    def __init__(self):
        self.val = 0.0
        self.iter = 0

    def update(self, x):
        self.val = (self.val * self.iter + x) / (self.iter + 1)
        self.iter += 1
    
    def __str__(self):
        if self.iter == 0:
            return '-'
        return f'{self.val:.3f}'


class Trainer(object):
    def __init__(self):
        self.net = model.Net()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.seg_criterion = model.seg_criterion
        self.tag_criterion = model.tag_criterion

        self.msg = None
        self.pbar = None

    def __train(self):
        self.net.train()
        self.msg.update({
            'loss': RunningAvg(),
            'seg_loss': RunningAvg(),
            'tag_loss': RunningAvg(),
        })
        for inp_batch, hmp_var, kpt_batch in iter(self.trainloader):
            inp_var = Variable(inp_batch, requires_grad=True).cuda()
            hmp_var = Variable(hmp_var, requires_grad=False).cuda()
            kpt_var = Variable(kpt_batch, requires_grad=False).cuda()

            out_var = self.net(inp_var)
            seg_loss = self.seg_criterion(out_var[:, :-1, ...], hmp_var)
            tag_loss = self.tag_criterion(out_var[:, -1, ...], kpt_var)
            loss = seg_loss + tag_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.msg['loss'].update(loss.data[0])
            self.msg['seg_loss'].update(seg_loss.data[0])
            self.msg['tag_loss'].update(tag_loss.data[0])
            self.pbar.set_postfix(**self.msg)
            self.pbar.update(32)

    def __valid(self):
        pass

    def __vis(self):
        pass

    def fit(self, trainset=None, validset=None, visset=None, epochs=50):
        self.trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
        self.validloader = DataLoader(validset, batch_size=32, shuffle=False, num_workers=4)
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