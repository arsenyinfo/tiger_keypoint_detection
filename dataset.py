import os
from copy import deepcopy
from functools import lru_cache
from typing import Callable, Collection, Optional

import cv2
import numpy as np
import ujson as json
from glog import logger
from torch.utils.data import Dataset
from tqdm import tqdm

import aug


def _read_img(x: str):
    img = cv2.imread(x)
    if img is None:
        logger.warning(f'Can not read image {x} with OpenCV')
        raise IOError('')
    return img


def parse_labels(img_dir, labels_path, approx_img_size=512):
    with open(labels_path) as f:
        annotation = json.load(f)
    images = annotation['images'][:8]
    labels = annotation['annotations'][:8]

    for img, lbl in tqdm(zip(images, labels), desc='preparing data', total=len(images)):
        assert img['id'] == lbl['image_id']
        valid_kpts = 0
        img = _read_img(os.path.join(img_dir, img['filename']))
        w, h, _ = img.shape
        coeff = 1
        if min(w, h) > approx_img_size:
            coeff = approx_img_size / min(w, h)
            img = cv2.resize(img, dsize=None, fx=coeff, fy=coeff, interpolation=cv2.INTER_CUBIC)

        kpts = []
        _kpts = lbl['keypoints']
        for i in range(len(_kpts) // 3):
            ky, kx, kv = _kpts[i * 3] * coeff, _kpts[i * 3 + 1] * coeff, _kpts[i * 3 + 2]
            if kv > 1:
                valid_kpts += 1
            kpts.append((kx, ky, kv))

        # if not valid_kpts:
        #     continue
        yield img, kpts


@lru_cache(maxsize=1)
def get_kernel2d(shape: Collection[int]) -> np.ndarray:
    size = (int(min(shape[:2]) * 0.2) // 2) * 2 + 1
    offset = size // 2
    kernel = cv2.getGaussianKernel(size, 0)
    kernel2d = kernel * kernel.T
    kernel2d /= kernel2d[offset, offset]
    return kernel2d


def get_heatmap(points, shape, presence):
    kernel2d = get_kernel2d(shape)
    offset = kernel2d.shape[0] // 2

    height, width, _ = shape
    heatmaps = np.zeros((height, width, len(presence)))

    for i, (point, pres) in enumerate(zip(points, presence)):
        if not pres:
            continue
        w, h = point

        h1, h2 = max(0, h - offset), min(height, h + offset + 1)
        w1, w2 = max(0, w - offset), min(width, w + offset + 1)

        if kernel2d.shape != (h2 - h1, w2 - w1):
            k = kernel2d.copy()[offset - h + h1: offset - h + h2, offset - w + w1: offset - w + w2]
        else:
            k = kernel2d.copy()

        heatmaps[h1: h2, w1: w2, i] = k
    return heatmaps


def calc_presence(keypoints, img):
    h, w, _ = img.shape
    res = [1 if 0 <= y < h and 0 <= x < w
           else 0
           for x, y in keypoints]
    return np.array(res)


class TigerDataset(Dataset):
    def __init__(self,
                 imgs,
                 labels,
                 size: int,
                 transform_fn: Callable,
                 normalize_fn: Callable,
                 corrupt_fn: Optional[Callable] = None,
                 verbose=True):
        self.imgs = imgs
        self.labels = labels
        self.size = size
        self.verbose = verbose
        self.corrupt_fn = corrupt_fn
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        logger.info(f'Dataset has been created with {len(self.imgs)} samples')

    def _preprocess(self, img, mask):
        def transpose(x):
            return np.transpose(x, (2, 0, 1))

        return transpose(self.normalize_fn(img)), transpose(mask)

    def __len__(self):
        return len(self.imgs)

    def get_raw(self, idx):
        img, kpts = self.imgs[idx], self.labels[idx]
        dataset_pres = np.array([x[-1] > 0 for x in kpts])
        kpts = np.array([x[:-1] for x in kpts])
        presence = calc_presence(kpts, img) * dataset_pres
        kpts = kpts * presence.reshape(-1, 1)
        img, kpts = self.transform_fn(img, kpts)
        return img, kpts, presence

    def __getitem__(self, idx):
        img, kpts, presence = self.get_raw(idx)
        if self.corrupt_fn is not None:
            img = self.corrupt_fn(img)
        presence = presence * calc_presence(kpts, img)
        kpts = kpts * presence.reshape(-1, 1)
        mask = get_heatmap(kpts.astype('int16'), img.shape, presence)
        norm_img, mask = self._preprocess(img, mask)
        return {'img': img.astype('float32'),
                'mask': mask.astype('float32'),
                'kpts': kpts.astype('float32'),
                'flags': presence.astype('float32'),
                'norm_img': norm_img.astype('float32')}

    @staticmethod
    def from_config(config):
        config = deepcopy(config)
        img_path = config.get('img_dir')
        labels_path = config.get('labels_path')

        imgs, labels = zip(*parse_labels(img_path, labels_path))
        transform_fn = aug.get_transforms(size=config['size'], crop=config['crop'])
        normalize_fn = aug.get_normalize()
        corrupt_fn = aug.get_corrupt_function(config['corrupt'])
        verbose = config.get('verbose', True)

        return TigerDataset(imgs=imgs,
                            labels=labels,
                            size=config['size'],
                            corrupt_fn=corrupt_fn,
                            normalize_fn=normalize_fn,
                            transform_fn=transform_fn,
                            verbose=verbose)

