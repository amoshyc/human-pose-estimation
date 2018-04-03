from pose.dataset import mpii
from pose.model.trainer import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable


trainset = mpii.MPIIDataset('./mpii/images/', './mpii/mpii_annotations.json', mode='train')
validset = mpii.MPIIDataset('./mpii/images/', './mpii/mpii_annotations.json', mode='valid')

trainer = Trainer()
trainer.fit(trainset, validset)

