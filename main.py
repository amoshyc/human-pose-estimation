import pathlib
from datetime import datetime

import torch
from torchvision.utils import save_image
from torchvision import transforms

from mpii import MPIItrain, MPIIvalid, MPIIsmall
from model import PoseEstimator

import warnings
from tqdm import TqdmSynchronisationWarning
warnings.simplefilter("ignore", TqdmSynchronisationWarning)

ckpt_dir = pathlib.Path(f'./ckpt/{datetime.now():%m-%d %H:%M:%S}/')
ckpt_dir.mkdir(parents=True)
device = torch.device('cuda')

est = PoseEstimator(ckpt_dir, device)
est.fit(MPIIsmall)
