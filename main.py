import pathlib
from datetime import datetime

import torch
from torch.utils.data.dataset import Subset, ConcatDataset

from mpii import MPII
from model import PoseEstimator

mpii_dir = pathlib.Path('./mpii')
MPIItrain = MPII(mpii_dir, mode='train', img_size=(224, 224))
MPIIvalid = MPII(mpii_dir, mode='valid', img_size=(224, 224))
# MPIItrain = Subset(MPII(mpii_dir, mode='train', img_size=(224, 224)), list(range(1000)))
# MPIIvalid = Subset(MPII(mpii_dir, mode='valid', img_size=(224, 224)), list(range(200)))
MPIIvis = ConcatDataset([
    Subset(MPIItrain, list(range(10))),
    Subset(MPIIvalid, list(range(10)))
])

ckpt_dir = pathlib.Path(f'./ckpt/{datetime.now():%m-%d %H:%M:%S}/')
ckpt_dir.mkdir(parents=True)
device = torch.device('cuda')

est = PoseEstimator(ckpt_dir, device)
est.fit(MPIItrain, MPIIvalid, MPIIvis)
