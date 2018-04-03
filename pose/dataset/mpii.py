import json
import pathlib

import numpy as np
from PIL import Image

from . import util

class MPIIDataset(object):
    def __init__(self,
                 image_dir=None,
                 json_path=None,
                 img_size=(256, 256),
                 hmp_size=(256, 256),
                 batch_size=2,
                 mode='train',):
        self.image_dir = pathlib.Path(image_dir).resolve()
        self.json_path = pathlib.Path(json_path).resolve()
        self.img_size = img_size
        self.hmp_size = hmp_size

        assert self.image_dir.exists()
        assert self.json_path.exists()

        is_valid = (mode != 'train')
        with open(json_path, 'r') as f:
            self.anno = [
                x for x in json.load(f) if x['isValidation'] == is_valid
            ]

    def __len__(self):
        return len(self.anno)

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
        jts = jts[:, :, [1, 0, 2]]
        jts[:, :, 0] = jts[:, :, 0] / h # normalized to 0~1
        jts[:, :, 1] = jts[:, :, 1] / w # normalized to 0~1

        hmp = np.zeros((16, *self.hmp_size), dtype=np.float32)
        for i in range(16):
            rs = np.round(jts[:, i, 0] * self.hmp_size[0]).astype(np.int32)
            cs = np.round(jts[:, i, 1] * self.hmp_size[1]).astype(np.int32)
            vs = jts[:, i, 2] == 1
            for r, c in zip(rs[vs], cs[vs]):
                rr, cc, g = util.gaussian2d((r, c), (1, 1), shape=self.hmp_size)
                hmp[i, rr, cc] = np.maximum(hmp[i, rr, cc], g / g.max())

        return img, hmp, jts


if __name__ == '__main__':
    ds = MPIIDataset('../mpii/images/', '../mpii/mpii_annotations.json')

    for idx in [222, 1100, 1600]:
        img, hmp, jts = ds[idx]
        util.visualize_hmp(img, hmp, f'{idx:04d}.png')
        util.visualize_jts(img, jts, f'{idx:04d}gt.png')

    
    