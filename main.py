from tqdm import tqdm
from pose.dataset import mpii
from pose.ae import model

model = model._make_ae()
model.summary()

