import json
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F


class COCOKeypoint:
    def __init__(self, img_dir, anno_path, img_size=(256, 256)):
        self.img_dir = pathlib.Path(img_dir)
        self.img_size = img_size

        with open(anno_path) as f:
            data = json.load(f)['annotations']

        self.annos = defaultdict(list)
        for anno in data:
            img_id = anno['image_id']
            kpt = anno['keypoints']
            box = anno['bbox']
            if anno['num_keypoints']:
                self.annos[img_id].append((kpt, box))
        self.keys = list(self.annos.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_id = self.keys[idx]
        img_path = self.img_dir / f'{img_id:012d}.jpg'
        img = Image.open(img_path)
        srcW, srcH = img.size
        dstH, dstW = self.img_size
        img = img.resize((dstW, dstH))

        n_people = len(self.annos[img_id])
        kpts = np.zeros((n_people * 17, 2), dtype=np.float32)
        viss = np.zeros((n_people * 17, ), dtype=np.int32)
        tags = np.zeros((n_people * 17, ), dtype=np.int32)
        boxs = np.zeros((n_people, 4), dtype=np.float32)
        for tag, (kpt, box) in enumerate(self.annos[img_id]):
            kpt = np.float32(kpt).reshape(-1, 3)
            vis = kpt[:, 2]
            kpt = kpt[:, [0, 1]]
            s, t = tag * 17, (tag + 1) * 17
            kpts[s:t] = kpt
            viss[s:t] = vis
            tags[s:t] = tag
            boxs[tag] = np.float32(box)

        norm_term = np.float32([dstW / srcW, dstH / srcH])
        kpts = np.round(kpts * norm_term)
        norm_term = np.hstack([norm_term, norm_term])
        boxs = np.round(boxs * norm_term)

        img = F.to_tensor(img)
        kpts = torch.from_numpy(kpts)
        viss = torch.from_numpy(viss)
        tags = torch.from_numpy(tags)
        boxs = torch.from_numpy(boxs)
        return img, kpts, viss, tags, boxs

    @staticmethod
    def collate_fn(batch):
        img_batch = torch.stack([datum[0] for datum in batch], dim=0)
        kpt_batch = list([datum[1] for datum in batch])
        vis_batch = list([datum[2] for datum in batch])
        tag_batch = list([datum[3] for datum in batch])
        box_batch = torch.stack([datum[4] for datum in batch], dim=0)
        return img_batch, kpt_batch, vis_batch, tag_batch, box_batch

    @staticmethod
    def plot(img, kpts, viss, tags, boxs):
        kpts = kpts.numpy()
        viss = viss.numpy()
        tags = tags.numpy()
        boxs = boxs.numpy()
        n_person = len(boxs)

        fig, ax = plt.subplots(dpi=100)
        ax.imshow(F.to_pil_image(img))

        # constants
        index = dict((x, i) for i, x in enumerate([
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle'
        ]))
        colors = plt.cm.tab20(np.linspace(0.0, 1.0, 20))
        colors = [mpl.colors.to_hex(c) for c in colors]
        edges = [
            # face
            ('left_ear', 'left_eye'),
            ('left_eye', 'nose'),
            ('left_eye', 'right_eye'),
            ('right_eye', 'nose'),
            ('right_eye', 'right_ear'),
            # upper body
            ('left_eye', 'left_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_eye', 'right_shoulder'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            # torso
            ('left_shoulder', 'left_hip'),
            ('left_hip', 'right_hip'),
            ('right_hip', 'right_shoulder'),
            ('right_shoulder', 'left_shoulder'),
            # lower body
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]

        for i in range(n_person):
            person_kpt = kpts[tags == i]
            person_vis = viss[tags == i]
            person_box = boxs[i]
            c = colors[i % 20]
            # edges
            for (s, t) in edges:
                s, t = index[s], index[t]
                if person_vis[s] and person_vis[t]:
                    pos = person_kpt[[s, t]]
                    ax.plot(pos[:, 0], pos[:, 1], c=c)
            # keypoints, 0: invisible, 1:occulation, 2:visible
            person_vis = person_vis >= 1
            person_kpt = person_kpt[person_vis]
            ax.plot(person_kpt[:, 0], person_kpt[:, 1], '.', c=c)
            # bbox
            x, y, w, h = person_box
            rect = mpl.patches.Rectangle((x, y), w, h, fill=False, ec=c, lw=1)
            ax.add_patch(rect)

        ax.axis('off')
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    img_dir = '/store/COCO/val2017/'
    anno_path = '/store/COCO/annotations/person_keypoints_val2017.json'
    ds = COCOKeypoint(img_dir, anno_path)

    img, kpts, viss, tags, boxs = ds[2]
    ds.plot(img, kpts, viss, tags, boxs)

    # loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=COCOKeypoint.collate_fn)
    # for img_batch, kpt_batch, vis_batch, tag_batch, box_batch in iter(loader):
    #     print(img_batch.size())
    #     print(len(kpt_batch))
    #     print(len(tag_batch))
    #     print(box_batch.size())
    #     break
