from pose.dataset import mpii
from pose.model.trainer import Trainer
from torch.utils.data import DataLoader

ds = mpii.MPIIDataset('./mpii/images/', './mpii/mpii_annotations.json', mode='train')
# validset = mpii.MPIIDataset('./mpii/images/', './mpii/mpii_annotations.json', mode='valid')

# trainer = Trainer()
# trainer.fit(trainset, validset)

dl = DataLoader(ds, batch_size=5, shuffle=False, num_workers=1)
dliter = iter(dl)

img_batch, hmp_batch, jts_batch = next(dliter)
print(img_batch.size())
print(hmp_batch.size())
print(jts_batch)