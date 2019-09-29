import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_fcn.tasks.grid_regression import GridRegressionHead
from pytorch_fcn.data.util import Dispenser
from pytorch_fcn.data.util import JointDispenser


class ContinuousGridRegression:

    def __init__(self, net, source_dataset, target_dataset, source_val_dataset, target_val_dataset, *, batch_size, crop_size=256, top=0, left=0, bottom=1024, right=2048, normalize=True, name='gridregression-cont'):
        self.net = net
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_val_dataset = source_val_dataset
        self.target_val_dataset = target_val_dataset
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.name = name
        self.normalize = normalize

        self.loss_fn = nn.MSELoss()
        self.create_datasets()
        self.create_head()

    def create_datasets(self):
        crop_transform = self.net.transform
        loaders = []
        for dataset in [self.source_dataset, self.target_dataset]:
            grid_dataset = ContinuousGridRegressionWrapper(
                dataset,
                crop_size=self.crop_size,
                top=self.top,
                left=self.left,
                bottom=self.bottom,
                right=self.right,
                normalize=self.normalize,
                crop_transform=crop_transform
            )
            loader = DataLoader(
                grid_dataset,
                batch_size=self.batch_size // 2,
                shuffle=True,
                num_workers=4,
                drop_last=True
            )
            loaders.append(loader)
        self.train_dispenser = JointDispenser(*loaders)
        val_loaders = []
        for dataset in [self.source_val_dataset, self.target_val_dataset]:
            grid_dataset = ContinuousGridRegressionWrapper(
                dataset,
                crop_size=self.crop_size,
                top=self.top,
                left=self.left,
                bottom=self.bottom,
                right=self.right,
                normalize=self.normalize,
                crop_transform=crop_transform
            )
            loader = DataLoader(
                grid_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=4
            )
            val_loaders.append(loader)
        self.val_loaders = {
            'source': val_loaders[0],
            'target': val_loaders[1],
        }

    def create_head(self):
        self.head = GridRegressionHead(self.net.out_dim)
        self.net.attach_head(self.name, self.head)

    def _predict_batch(self, im):
        im = im.cuda()
        preds = self.net(im, task=self.name)
        preds = preds.view(-1, 2)
        return preds

    def step(self):
        im, label = self.train_dispenser.next_batch()
        label = label.cuda()
        preds = self._predict_batch(im)
        loss = self.loss_fn(preds, label)
        return loss

    def eval(self):
        self.net.eval()
        results = {}
        for domain, loader in self.val_loaders.items():
            errs = []
            for im, label in loader:
                with torch.no_grad():
                    label = label.cuda()
                    preds = self._predict_batch(im)
                    err = self.loss_fn(preds, label)
                    errs.append(err.item())
            avg_err = np.mean(errs)
            logging.info(f'    {self.name}.{domain}: {avg_err}')
            results[f'{self.name}.{domain}'] = avg_err
        self.net.train()
        return results


class ContinuousGridRegressionWrapper:

    def __init__(self, dataset, *, crop_size=256, top=0, left=0, bottom=1024, right=2048, transform=None, crop_transform=None, normalize=True):
        self.dataset = dataset
        self.crop_size = crop_size
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.transform = transform
        self.crop_transform = crop_transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im = self.dataset[index][0]
        if self.transform is not None:
            im = self.transform(im)
        if self.right is None:
            self.right = im.size[0]
        if self.bottom is None:
            self.bottom = im.size[1]
        x_max = (self.right - self.left) - self.crop_size
        y_max = (self.bottom - self.top) - self.crop_size
        x = random.randint(0, x_max)
        y = random.randint(0, y_max)
        crop = self.crop(im, self.left + x, self.top + y)
        if self.crop_transform is not None:
            crop = self.crop_transform(crop)
        if not self.normalize:
            max_dim = max(x_max, y_max)
            x_max = y_max = max_dim
        target = torch.Tensor([x / x_max, y / y_max])
        return crop, target

    def crop(self, im, x, y):
        left = x
        top = y
        right = x + self.crop_size
        bottom = y + self.crop_size
        im = im.crop((left, top, right, bottom))
        return im
