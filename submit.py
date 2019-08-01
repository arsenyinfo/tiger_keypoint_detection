import json
from glob import glob
from typing import Sequence

import cv2
import numpy as np
import torch
from albumentations import PadIfNeeded
from glog import logger
from torch.serialization import load
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aug import get_normalize
from metrics import subpixel_argmax2d


class TigerTestDataset(Dataset):
    def __init__(self,
                 imgs: Sequence[str],
                 ):
        self.imgs = imgs
        self.normalize_fn = get_normalize()
        self.approx_img_size = 384
        self.pad_params = dict(border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        logger.info(f'Dataset has been created with {len(self.imgs)} samples')

    def _preprocess(self, img):
        def transpose(x):
            return np.transpose(x, (2, 0, 1))

        return transpose(self.normalize_fn(img))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        fpath = self.imgs[idx]
        img = cv2.imread(fpath)

        old_shape = img.shape
        h, w, _ = old_shape
        coeff = 1
        if min(w, h) > self.approx_img_size:
            coeff = self.approx_img_size / min(w, h)
            img = cv2.resize(img, dsize=None, fx=coeff, fy=coeff, interpolation=cv2.INTER_CUBIC)
        new_shape = img.shape
        h, w, _ = new_shape

        block_size = 16
        pad = PadIfNeeded(min_height=(h // block_size + 1) * block_size, min_width=(w // block_size + 1) * block_size,
                          **self.pad_params)
        img = pad(image=img)['image']

        img = self._preprocess(img)
        return {'img': img.astype('float32'),
                'file': fpath,
                'coeff': coeff,
                'new_shape': new_shape,
                'old_shape': old_shape}


def get_keypoints(heatmaps):
    heatmaps = heatmaps[0].cpu().numpy()
    res = []
    for heatmap in heatmaps:
        res.append(subpixel_argmax2d(heatmap))
    return np.array(res)


def make_annotation(heatmaps, flags, params: dict):
    kpts = get_keypoints(heatmaps)
    flags = flags.cpu().numpy() > .5
    kpts = np.hstack((kpts, flags.reshape(-1, 1)))
    print(kpts.max(axis=0), params)


def main(model_path: str = 'baseline/model.pt',
         files_pattern='/home/arseny/datasets/atrw/test/*.jpg',
         output='result.json'):
    model = load(model_path)
    files = glob(files_pattern)
    dataset = TigerTestDataset(imgs=files)
    dataloader = DataLoader(dataset, drop_last=False, batch_size=1, num_workers=0, shuffle=False)
    dataloader = iter(dataloader)

    result = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs = batch.pop('img').to('cuda')
            heatmaps, flags = map(torch.sigmoid, model(imgs))
            annotation = make_annotation(heatmaps, flags, batch)

    with open(output, 'w') as out:
        json.dump(result, out)


if __name__ == '__main__':
    main()
