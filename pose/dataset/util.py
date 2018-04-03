import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from PIL import Image
import skimage
from skimage import draw
from skimage import feature
from skimage import transform


def gaussian2d(mu, sigma, shape=None):
    """Generate 2d gaussian distribution coordinates and values.

    Parameters
    --------------
    mu: tuple of int
        Coordinates of center, (mu_r, mu_c)
    sigma: tuple of int
        Intensity of the distribution, (sigma_r, sigma_c)
    shape: tuple of int, optional
        Image shape which is used to determine the maximum extent
        pixel coordinates, (r, c)

    Returns
    --------------
    rr, cc: (N,) ndarray of int
        Indices of pixels that belong to the distribution
    gaussian: (N, ) ndarray of float
        Values of corresponding position. Ranges from 0.0 to 1.0.

    .. warning::

        This function does NOT support mu, sigma as double.
    """
    mu_r, mu_c = mu
    sigma_r, sigma_c = sigma

    R, C = 6 * sigma_r + 1, 6 * sigma_c + 1
    r = np.arange(-3 * sigma_r, +3 * sigma_r + 1) + mu_r
    c = np.arange(-3 * sigma_c, +3 * sigma_c + 1) + mu_c
    if shape:
        r = np.clip(r, 0, shape[0] - 1)
        c = np.clip(c, 0, shape[1] - 1)

    coef_r = 1 / (sigma_r * np.sqrt(2 * np.pi))
    coef_c = 1 / (sigma_c * np.sqrt(2 * np.pi))
    exp_r = -1 / (2 * (sigma_r**2)) * ((r - mu_r)**2)
    exp_c = -1 / (2 * (sigma_c**2)) * ((c - mu_c)**2)

    gr = coef_r * np.exp(exp_r)
    gc = coef_c * np.exp(exp_c)
    g = np.outer(gr, gc)

    r = r.reshape(R, 1)
    c = c.reshape(1, C)
    rr = np.broadcast_to(r, (R, C))
    cc = np.broadcast_to(c, (R, C))
    return rr.ravel(), cc.ravel(), g.ravel()


def visualize_jts(img, jts, name):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    cm = plt.cm.tab20
    n_people = len(jts)
    img_h, img_w = img.shape[:2]

    for i in range(n_people):
        rs = np.round(jts[i, :, 0] * img_h).astype(np.int32)
        cs = np.round(jts[i, :, 1] * img_w).astype(np.int32)
        vs = jts[i, :, 2] == 1 # visible
        ax.plot(cs[vs], rs[vs], '.', color=cm(i / 20))

    lines = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8],
             [10, 11], [11, 12], [13, 14], [14, 15], [7, 12], [7, 13], [8, 9]]
    for i in range(n_people):
        rs = np.round(jts[i, :, 0] * img_h).astype(np.int32)
        cs = np.round(jts[i, :, 1] * img_w).astype(np.int32)
        vs = jts[i, :, 2] == 1  # visible
        for j1, j2 in lines:
            if vs[j1] and vs[j2]:
                ax.plot(cs[[j1, j2]], rs[[j1, j2]], color=cm(i / 20))

    if not name.endswith('.png'):
        name += '.png'
    fig.savefig(name, bbox_inches='tight', pad_inches=0)


def visualize_hmp(img, hmp, name):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    cm = plt.cm.tab20
    img_h, img_w = img.shape[:2]

    for i in range(16):
        peaks = feature.peak_local_max(hmp[i], exclude_border=False)
        ax.plot(peaks[:, 1], peaks[:, 0], 'r.')

    if not name.endswith('.png'):
        name += '.png'
    fig.savefig(name, bbox_inches='tight', pad_inches=0)