import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from skimage import feature

from torchvision.utils import save_image
from torchvision import transforms

from mpii import MPIItrain, MPIIvalid

img, lbl, tag = MPIItrain[0]

img = transforms.ToPILImage()(img)
lbl = lbl.numpy()
tag = tag.numpy()

fig, ax = plt.subplots(dpi=100)
ax.imshow(img)
ax.axis('off')
for i in range(16):
    peaks = feature.peak_local_max(lbl[i], exclude_border=False)
    ax.plot(peaks[:, 1], peaks[:, 0], 'r.')
fig.tight_layout()
plt.show()
