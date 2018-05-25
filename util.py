import numpy as np


class RunningAverage(object):
    def __init__(self):
        super().__init__()
        self.iter = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = (self.avg * self.iter + x.item()) / (self.iter + 1)
        self.iter += 1

    def __str__(self):
        if self.iter == 0:
            return 'x'
        return f'{self.avg:.4f}'


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
