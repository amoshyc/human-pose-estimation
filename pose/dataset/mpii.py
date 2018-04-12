import json
import pathlib

import numpy as np
from PIL import Image
import torch

from . import util

class MPIIDataset(object):
    def __init__(self,
                 image_dir=None,
                 json_path=None,
                 img_size=(256, 256),
                 lbl_size=(256, 256),
                 batch_size=2,
                 mode='train',):
        self.image_dir = pathlib.Path(image_dir).resolve()
        self.json_path = pathlib.Path(json_path).resolve()
        self.img_size = img_size
        self.lbl_size = lbl_size

        assert self.image_dir.exists()
        assert self.json_path.exists()

        is_valid = (mode != 'train')
        with open(json_path, 'r') as f:
            self.anno = [
                x for x in json.load(f) if x['isValidation'] == is_valid
            ]

    def __len__(self):
        # return len(self.anno)
        return 2000

    def __getitem__(self, idx):
        anno = self.anno[idx]
        h = anno['img_height']
        w = anno['img_width']
        img_path = str(self.image_dir / anno['img_paths'])
        n_people = int(anno['numOtherPeople'] + 1)

        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(self.img_size)
        img = np.array(img) / 255

        jts = [anno['joint_self']]
        if n_people == 2:
            jts.append(anno['joint_others'])
        if n_people > 2:
            for jt in anno['joint_others']:
                jts.append(jt)

        jts = np.float32(jts) # (n_people, 16, 3)
        jts = jts[:, :, [2, 1, 0]] # [x, y, v] -> [v, r, c]
        jts[:, :, 1] = np.round(jts[:, :, 0] / h * self.lbl_size[0])
        jts[:, :, 2] = np.round(jts[:, :, 1] / w * self.lbl_size[1])
        jts = np.int32(jts)

        hmp = np.zeros((*self.lbl_size, 16), dtype=np.float32)
        for i in range(16):
            vs = jts[:, i, 0] == 1
            rs = jts[:, i, 1][vs]
            cs = jts[:, i, 2][vs]
            for r, c in zip(rs, vs):
                rr, cc, g = util.gaussian2d((r, c), (2, 2), shape=self.lbl_size)
                hmp[rr, cc, i] = np.maximum(hmp[i, rr, cc], g / g.max())

        kpt = np.zeros(self.lbl_size, dtype=np.int32) # a (label) mask
        for i in range(n_people):
            vs = jts[i, :, 0] == 1
            rs = jts[i, :, 1][vs]
            cs = jts[i, :, 2][vs]
            kpt[rs, cs] = (i + 1)

        return img, hmp, kpt

    def flow(self, batch_size):
        idx = 0
        img_batch = np.zeros((batch_size, *self.img_size, 3), dtype=np.float32)
        hmp_batch = np.zeros((batch_size, *self.lbl_size, 16), dtype=np.float32)
        kpt_batch = np.zeros((batch_size, *self.lbl_size), dtype=np.float32)

        while True:
            n_samples = len(self)
            indices = np.random.permutation(n_samples)
            for i in indices:
                img, hmp, kpt = self[i]
                img_batch[idx] = img
                hmp_batch[idx] = hmp
                kpt_batch[idx] = kpt
                if idx + 1 == batch_size:
                    yield img_bath, hmp_batch, kpt_batch
                idx = (idx + 1) % batch_size


if __name__ == '__main__':
    ds = MPIIDataset('../mpii/images/', '../mpii/mpii_annotations.json')

    
    