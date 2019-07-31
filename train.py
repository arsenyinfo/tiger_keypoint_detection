import os
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from glog import logger
from joblib import cpu_count
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.serialization import save
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TigerDataset
from metrics import subpixel_argmax2d
from models import TigerFPN


def iou_continuous_loss_with_logits(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    eps = 1e-6

    def _sum(x):
        return x.sum(-1).sum(-1)

    numerator = (_sum(y_true * y_pred) + eps)
    denominator = (_sum(y_true ** 2) + _sum(y_pred ** 2) - _sum(y_true * y_pred) + eps)
    return 1 - (numerator / denominator).mean()


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train: Optional[DataLoader],
                 val: Optional[DataLoader],
                 segm_loss_fn: Callable,
                 clf_loss_fn: Callable,
                 epochs: int = 200,
                 early_stop: int = 10,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[ReduceLROnPlateau] = None,
                 device: str = 'cuda:0',
                 checkpoint: str = './model.pt',
                 work_dir: str = '.'
                 ):
        self.epochs = epochs
        self.early_stop = early_stop
        self.model = model.to(device)
        self.device = device
        self.train = train
        self.val = val
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler if scheduler is not None else ReduceLROnPlateau(optimizer=self.optimizer,
                                                                                   verbose=True)
        self.clf_loss_fn = clf_loss_fn
        self.segm_loss_fn = segm_loss_fn
        self.current_metric = -np.inf
        self.last_improvement = 0
        self.work_dir = work_dir
        self.checkpoint = os.path.join(self.work_dir, checkpoint)
        self.tb_writer = SummaryWriter(logdir=work_dir)

    def to_device(self, x):
        if type(x) == dict:
            return {k: self.to_device(v) for k, v in x.items()}
        return x.to(self.device)

    @staticmethod
    def get_score(pred_mask, kpts):
        res = []
        for pred, gt in zip(pred_mask.detach().cpu().numpy(), kpts.cpu().numpy()):
            for i in range(len(gt)):
                pred_layer = pred[i]
                kpt = gt[i]
                coord = subpixel_argmax2d(pred_layer)
                if np.all(np.isnan(coord)):
                    continue
                y, x = coord
                res.append(np.abs(kpt - np.array([x, y])))
        return np.mean(res)

    def write_from_batch(self, mask, pred_mask, img, tag: str, n: int):
        pred_mask = torch.sigmoid(pred_mask[0]).detach().cpu().numpy().max(axis=0)
        mask = mask[0].detach().cpu().numpy().max(axis=0)

        mask = np.hstack((pred_mask, mask))
        assert 0 <= mask.max() <= 1, mask.max()
        assert 1 <= img.max() <= 255
        img = np.hstack((img.astype('float32') / 255, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        img = np.expand_dims(img, 0)
        self.tb_writer.add_images(tag=tag, img_tensor=img, dataformats='NHWC', global_step=n)

    def _train_epoch(self, n_epoch):
        self.model.train(True)
        segm_losses, clf_losses, oks = [], [], []

        train = tqdm(self.train, desc=f'training, epoch {n_epoch}')
        for i, inputs in enumerate(train):
            inputs = self.to_device(inputs)
            img = inputs['norm_img']
            flags = inputs['flags']
            kpts = inputs['kpts']
            mask = inputs['mask']

            self.optimizer.zero_grad()
            pred_mask, pred_flags = self.model(img)

            loss_mask = self.segm_loss_fn(pred_mask, mask).mean()
            loss_flags = self.clf_loss_fn(pred_flags, flags)
            loss = loss_mask + loss_flags
            loss.backward()

            self.write_from_batch(mask, pred_mask, inputs['img'][0].cpu().numpy(), 'train', n_epoch)

            segm_losses.append(loss_mask.item())
            clf_losses.append(loss_flags.item())
            metric = self.get_score(pred_mask, kpts)
            oks.append(metric)

            train.set_postfix(segm_loss=f'{loss_mask.item():.3f}',
                              clf_loss=f'{loss_flags.item():.3f}',
                              oks=f'{metric:.3f}')
            self.optimizer.step()

        train.close()
        return segm_losses, clf_losses, oks

    def _val_epoch(self, n_epoch):
        self.model.train(False)
        segm_losses, clf_losses, oks = [], [], []

        val = tqdm(self.val, desc=f'validating, epoch {n_epoch}')
        with torch.no_grad():
            for i, inputs in enumerate(val):
                inputs = self.to_device(inputs)
                img = inputs['norm_img']
                flags = inputs['flags']
                kpts = inputs['kpts']
                mask = inputs['mask']

                pred_mask, pred_flags = self.model(img)
                self.write_from_batch(mask, pred_mask, inputs['img'][0].cpu().numpy(), 'val', n_epoch)

                loss_mask = self.segm_loss_fn(pred_mask, mask).mean()
                loss_flags = self.clf_loss_fn(pred_flags, flags)
                segm_losses.append(loss_mask.item())
                clf_losses.append(loss_flags.item())
                metric = self.get_score(pred_mask, kpts)
                oks.append(metric)
                val.set_postfix(segm_loss=f'{loss_mask.item():.3f}',
                                clf_loss=f'{loss_flags.item():.3f}',
                                oks=f'{metric:.3f}')
        val.close()
        return segm_losses, clf_losses, oks

    def fit_one_epoch(self, n_epoch):
        segm_losses, clf_losses, oks = self._train_epoch(n_epoch)
        val_segm_losses, val_clf_losses, val_oks = self._val_epoch(n_epoch)

        train_segm_loss = np.mean(segm_losses)
        val_segm_loss = np.mean(val_segm_losses)
        train_clf_loss = np.mean(clf_losses)
        val_clf_loss = np.mean(val_clf_losses)
        oks = np.mean(oks)
        val_oks = np.mean(val_oks)

        msg = f'Epoch {n_epoch}: ' \
              f'train segm loss is {train_segm_loss:.3f}, ' \
              f'train clf loss  {train_clf_loss:.3f}, ' \
              f'train oks  {oks:.3f}, ' \
              f'val segm loss is {val_segm_loss:.3f}, ' \
              f'val clf loss  {val_clf_loss:.3f}, ' \
              f'val oks  {val_oks:.3f}, '
        logger.info(msg)

        self.scheduler.step(metrics=val_segm_loss + val_clf_loss, epoch=n_epoch)

        metric = -train_segm_loss - train_clf_loss
        if metric > self.current_metric:
            self.current_metric = metric
            self.last_improvement = n_epoch
            save(self.model, f=self.checkpoint)
            logger.info(f'Best model has been saved at {n_epoch}, metric is {metric:.4f}')
        else:
            if self.last_improvement + self.early_stop < n_epoch:
                return True, (train_segm_loss, train_clf_loss, oks,
                              val_segm_loss, val_clf_loss, val_oks)

        return False, (train_segm_loss, train_clf_loss, oks,
                       val_segm_loss, val_clf_loss, val_oks)

    def fit(self, start_epoch: int):
        for i in range(self.epochs):
            finished, metrics = self.fit_one_epoch(i + start_epoch)
            train_segm_loss, train_clf_loss, train_oks, val_segm_loss, val_clf_loss, val_oks = metrics
            for name, scalar in (('train_segm_loss', train_segm_loss),
                                 ('train_clf_loss', train_clf_loss),
                                 ('train_oks', train_oks),
                                 ('val_segm_loss', val_segm_loss),
                                 ('val_clf_loss', val_clf_loss),
                                 ('val_oks', val_oks)
                                 ):
                self.tb_writer.add_scalar(name, scalar, global_step=i)
            if finished:
                return i
        return self.epochs


def make_dataloaders(train_cfg, val_cfg, batch_size, multiprocessing=False):
    train = TigerDataset.from_config(train_cfg)
    val = TigerDataset.from_config(val_cfg)

    shared_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': cpu_count() if multiprocessing else 0}

    train = DataLoader(train, drop_last=True, **shared_params)
    val = DataLoader(val, drop_last=False, **shared_params)
    return train, train


def update_config(config, params):
    for k, v in params.items():
        *path, key = k.split('.')
        conf = config
        for p in path:
            if p not in conf:
                logger.error(f'Overwriting non-existing attribute {k} = {v}')
            conf = conf[p]
        logger.info(f'Overwriting {k} = {v} (was {conf.get(key)})')
        conf[key] = v


def fit(parallel=False, **kwargs):
    with open('config.yaml') as cfg:
        config = yaml.load(cfg)
    update_config(config, kwargs)
    work_dir = config['name']
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, 'config.yaml'), 'w') as out:
        yaml.dump(config, out)

    train, val = make_dataloaders(config['train'], config['val'], config['batch_size'], multiprocessing=parallel)
    model = TigerFPN()
    # model = DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    trainer = Trainer(model=model,
                      train=train,
                      val=val,
                      clf_loss_fn=F.binary_cross_entropy_with_logits,
                      segm_loss_fn=iou_continuous_loss_with_logits,
                      # segm_loss_fn=F.binary_cross_entropy_with_logits,
                      work_dir=work_dir,
                      optimizer=optimizer,
                      scheduler=ReduceLROnPlateau(factor=.2, patience=10, optimizer=optimizer),
                      device='cuda:0',
                      epochs=config['n_epochs'],
                      early_stop=config['early_stop']
                      )
    trainer.fit(start_epoch=0)


if __name__ == '__main__':
    Fire(fit)
