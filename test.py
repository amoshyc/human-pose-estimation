import pathlib
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')

import torch
from torch.utils.data.dataset import Subset, ConcatDataset

from coco import COCOKeypoint
from model import PoseEstimator

COCOTrain = COCOKeypoint(
    '/store/COCO/train2017/',
    '/store/COCO/annotations/person_keypoints_train2017.json')
COCOValid = COCOKeypoint(
    '/store/COCO/val2017/',
    '/store/COCO/annotations/person_keypoints_val2017.json')
COCOVis = ConcatDataset([
    Subset(COCOTrain, list(range(40))),
    Subset(COCOValid, list(range(40)))
])

log_dir = pathlib.Path(f'./log/{datetime.now():%m-%d %H:%M:%S}/')
log_dir.mkdir(parents=True)
device = torch.device('cuda')

est = PoseEstimator(log_dir, device)
est.fit(COCOTrain, COCOValid, COCOVis)
