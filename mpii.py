import json
import random
import shutil
import pathlib
import urllib.request

import numpy as np
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm
from skimage import measure

import torch
from torchvision import datasets
from torchvision import transforms

import util


class MPII(object):
    def __init__(self, root_dir, mode='train', img_size=(256, 256)):
        super().__init__()
        self.root_dir = pathlib.Path(root_dir) / mode
        self.img_size = img_size  # H, W
        with (self.root_dir / 'anno.json').open() as f:
            self.anno = json.load(f)

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        anno = self.anno[idx]
        img_path = self.root_dir / anno['filename']
        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        img = img.resize(self.img_size[::-1])
        lbl = np.zeros((16, *self.img_size), dtype=np.float32)
        tag = np.zeros((1, *self.img_size), dtype=np.uint8)

        # Draw Gaussian & tag
        for pid, person in enumerate(anno['people']):
            for jid, (x, y) in person['joints'].items():
                jid = int(jid)
                r = min(round(y / H * self.img_size[0]), self.img_size[0] - 1)
                c = min(round(x / W * self.img_size[1]), self.img_size[1] - 1)
                tag[0, r, c] = pid + 1
                rr, cc, g = util.gaussian2d(
                    [r, c], [3, 3], shape=self.img_size)
                lbl[jid, rr, cc] = np.maximum(lbl[jid, rr, cc], g / g.max())

        # Convert to tensor
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean, std)(img)
        lbl = torch.tensor(lbl)
        tag = torch.tensor(tag)
        return img, lbl, tag


class MPIIParse(object):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = pathlib.Path(root_dir)
        self.train_dir = self.root_dir / 'train'
        self.valid_dir = self.root_dir / 'valid'

        print('Extracting', end='...')
        self.extract()
        print('OK')

        print('Spliting', end='...')
        self.split()
        print('OK')

    def extract(self):
        shutil.unpack_archive(
            str(self.root_dir / 'mpii.tar.gz'), str(self.root_dir))
        shutil.unpack_archive(
            str(self.root_dir / 'anno.zip'), str(self.root_dir))

    def split(self):
        mat_path = 'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
        anno = self._mat2json(str(self.root_dir / mat_path))
        with (self.root_dir / 'anno_all.json').open('w') as f:
            json.dump(anno, f)

        n_imgs = len(anno)
        random.shuffle(anno)
        pivot = n_imgs // 5 * 4
        train_anno = anno[:pivot]
        valid_anno = anno[pivot:]

        self.train_dir.mkdir(exist_ok=True, parents=True)
        self.valid_dir.mkdir(exist_ok=True, parents=True)
        with (self.train_dir / 'anno.json').open('w') as f:
            json.dump(train_anno, f)
        with (self.valid_dir / 'anno.json').open('w') as f:
            json.dump(valid_anno, f)
        for anno in train_anno:
            img_path = self.root_dir / 'images' / anno['filename']
            shutil.move(str(img_path), str(self.train_dir))
        for anno in valid_anno:
            img_path = self.root_dir / 'images' / anno['filename']
            shutil.move(str(img_path), str(self.valid_dir))

    def _mat2json(self, mat_file):
        mat = loadmat(mat_file)['RELEASE']
        indices = mat['img_train'][0, 0][0].astype(bool)
        annos = mat['annolist'][0, 0][0]

        result = []
        for anno in annos[indices]:
            data = {'filename': anno['image']['name'][0, 0][0], 'people': []}
            try:
                people = anno['annorect']['annopoints'][0]
            except:
                continue
            for person in people:
                if len(person) == 0:
                    continue
                points = person['point'][0, 0]
                joints_id = points['id'][0]
                joints_xs = points['x'][0]
                joints_ys = points['y'][0]
                joints = dict()
                for jid, x, y in zip(joints_id, joints_xs, joints_ys):
                    joints[int(jid[0, 0])] = [float(x[0, 0]), float(y[0, 0])]

                try:
                    joints_vis = [bool(v) for v in points['is_visible'][0]]
                except:
                    joints_vis = [False for _ in range(16)]

                data['people'].append({
                    'visibility': joints_vis,
                    'joints': joints
                })

            result.append(data)
        return result
