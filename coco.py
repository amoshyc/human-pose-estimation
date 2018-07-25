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
    def __init__(self, img_dir, anno_path, img_size=(256, 256), lbl_size=(128, 128)):
        self.img_dir = pathlib.Path(img_dir)
        self.img_size = img_size
        self.lbl_size = lbl_size

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
        img = Image.open(img_path).convert('RGB')
        srcW, srcH = img.size
        img = img.resize(self.img_size[::-1])

        n_people = len(self.annos[img_id])
        kpts = np.zeros((n_people * 17, 2), dtype=np.float32)   # [0, 1]
        viss = np.zeros((n_people * 17, ), dtype=np.int32)      # 0 or 1
        tags = np.zeros((n_people * 17, ), dtype=np.int32)      # Z+
        boxs = np.zeros((n_people, 4), dtype=np.float32)        # [0, 1]
        for tag, (kpt, box) in enumerate(self.annos[img_id]):
            kpt = np.float32(kpt).reshape(-1, 3)
            vis = kpt[:, 2]
            kpt = kpt[:, [1, 0]]                                # xy -> rc
            c, r, w, h = box
            s, t = tag * 17, (tag + 1) * 17
            kpts[s:t] = kpt
            viss[s:t] = vis
            tags[s:t] = tag
            boxs[tag] = np.float32([r, c, h, w])

        kpts = kpts / np.float32([srcH, srcW])
        boxs = boxs / np.float32([srcH, srcW, srcH, srcW])

        lblH, lblW = self.lbl_size
        lbl = np.zeros((17, lblH, lblW), dtype=np.float32)
        for k in range(17):
            vis = viss[k::17] > 0
            kpt = kpts[k::17]
            vis_kpts = kpt[vis]
            vis_kpts = vis_kpts * np.float32([lblH, lblW])
            vis_kpts = np.round(vis_kpts).astype(np.int32)
            for (r, c) in vis_kpts:
                r, c = int(r), int(c)
                rr, cc, g = self.gaussian([r, c], [3, 3], shape=(lblH, lblW))
                lbl[k, rr, cc] = np.maximum(lbl[k, rr, cc], g / g.max())

        img = F.to_tensor(img).float()          # (3, lblH, lblW) float
        lbl = torch.from_numpy(lbl).float()     # (17, lblH, lblW) float
        kpts = torch.from_numpy(kpts).float()   # (n_people * 17, 2) float
        viss = torch.from_numpy(viss).long()    # (n_poeple * 17,) int
        tags = torch.from_numpy(tags).long()    # (n_people * 17,) int
        boxs = torch.from_numpy(boxs).float()   # (n_people, 4) float
        return img, lbl, kpts, viss, tags, boxs

    @staticmethod
    def gaussian(mu, sigma, shape=None):
        (r, c), (sr, sc), (H, W) = mu, sigma, shape
        rr = np.arange(r - 3 * sr, r + 3 * sr + 1)
        cc = np.arange(c - 3 * sc, c + 3 * sc + 1)
        rr = rr[(rr >= 0) & (rr < H)]
        cc = cc[(cc >= 0) & (cc < W)]
        gr = np.exp(-0.5 * ((rr - r) / sr)**2) / (np.sqrt(2 * np.pi) * sr)
        gc = np.exp(-0.5 * ((cc - c) / sc)**2) / (np.sqrt(2 * np.pi) * sc)
        g = np.outer(gr, gc).ravel()
        R, C = len(rr), len(cc)
        rr = np.broadcast_to(rr.reshape(R, 1), (R, C)).ravel()
        cc = np.broadcast_to(cc.reshape(1, C), (R, C)).ravel()
        return rr, cc, g

    @staticmethod
    def collate_fn(batch):
        img_batch = torch.stack([datum[0] for datum in batch], dim=0)
        lbl_batch = torch.stack([datum[1] for datum in batch], dim=0)
        kpt_batch = list([datum[2] for datum in batch])
        vis_batch = list([datum[3] for datum in batch])
        tag_batch = list([datum[4] for datum in batch])
        box_batch = list([datum[5] for datum in batch])
        return img_batch, lbl_batch, kpt_batch, vis_batch, tag_batch, box_batch

    @staticmethod
    def plot(tensor_img, tensor_kpts, tensor_viss, tensor_tags, tensor_boxs, ax=None):
        img = F.to_pil_image(tensor_img)
        kpts = tensor_kpts.numpy()
        viss = tensor_viss.numpy()
        tags = tensor_tags.numpy()
        boxs = tensor_boxs.numpy()

        imgW, imgH = img.size
        kpts = kpts * np.float32([imgW, imgH])
        boxs = boxs * np.float32([imgW, imgH, imgW, imgH])
        kpts = np.round(kpts)
        boxs = np.round(boxs)

        n_people = len(boxs)
        ax.imshow(img)
        ax.axis('off')

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

        for i in range(n_people):
            person_kpt = kpts[tags == i]
            person_vis = viss[tags == i]
            person_box = boxs[i]
            c = colors[i % 20]
            # edges
            for (s, t) in edges:
                s, t = index[s], index[t]
                if person_vis[s] and person_vis[t]:
                    pos = person_kpt[[s, t]]
                    ax.plot(pos[:, 1], pos[:, 0], c=c)
            # keypoints, 0: invisible, 1:occulation, 2:visible
            person_vis = person_vis >= 1
            person_kpt = person_kpt[person_vis]
            ax.plot(person_kpt[:, 1], person_kpt[:, 0], '.', c=c)
            # bbox
            y, x, h, w = person_box
            rect = mpl.patches.Rectangle((x, y), w, h, fill=False, ec=c, lw=1)
            ax.add_patch(rect)


if __name__ == '__main__':
    img_dir = '/store/COCO/val2017/'
    anno_path = '/store/COCO/annotations/person_keypoints_val2017.json'
    ds = COCOKeypoint(img_dir, anno_path)

    img, lbl, kpts, viss, tags, boxs = ds[2]
    fig, ax = plt.subplots(1, 2, dpi=100)
    ds.plot(img, kpts, viss, tags, boxs, ax=ax[0])
    ax[1].imshow(lbl[5].numpy(), cmap='gray')
    plt.show()

    # loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=ds.collate_fn)
    # for img_batch, lbl_batch, kpt_batch, \
    #     vis_batch, tag_batch, box_batch in iter(loader):
    #     # print(img_batch.size())
    #     # print(lbl_batch.size())
    #     # print(len(kpt_batch))
    #     # print(len(tag_batch))
    #     # print(len(vis_batch))
    #     # print(len(box_batch))
    #     pass
