import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_fcn.data.util import Dispenser
from pytorch_fcn.data.util import JointDispenser


class Grid:

    def __init__(self, net, source_dataset, target_dataset, source_val_dataset, target_val_dataset, *, batch_size, name='grid'):
        self.net = net
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_val_dataset = source_val_dataset
        self.target_val_dataset = target_val_dataset
        self.batch_size = batch_size
        self.name = name

        self.loss_fn = nn.NLLLoss()
        self.create_datasets()
        self.create_head()

    def create_datasets(self):
        transform = transforms.Resize(1024)
        crop_transform = self.net.transform
        loaders = []
        for dataset in [self.source_dataset, self.target_dataset]:
            grid_dataset = GridWrapper(
                dataset,
                transform=transform,
                crop_transform=crop_transform
            )
            loader = DataLoader(
                grid_dataset,
                batch_size=self.batch_size // 2,
                shuffle=True,
                num_workers=4
            )
            loaders.append(loader)
        val_loaders = []
        for dataset in [self.source_val_dataset, self.target_val_dataset]:
            grid_dataset = GridWrapper(
                dataset,
                transform=transform,
                crop_transform=crop_transform
            )
            loader = DataLoader(
                grid_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=4
            )
            val_loaders.append(loader)
        self.train_dispenser = JointDispenser(*loaders)
        self.val_loaders = {
            'source': val_loaders[0],
            'target': val_loaders[1],
        }

    def create_head(self):
        self.head = GridHead(self.net.out_dim)
        self.net.attach_head(self.name, self.head)

    def _predict_batch(self, im):
        n, g, c, h, w = im.size()
        im = im.view(n * g, c, h, w).cuda()
        preds = self.net(im, task=self.name)
        preds = preds.view(n * g, 8)
        return preds

    def step(self):
        im, label = self.train_dispenser.next_batch()
        label = label.view(-1).cuda()
        preds = self._predict_batch(im)
        loss = self.loss_fn(preds, label)
        return loss

    def eval(self):
        self.net.eval()
        results = {}
        for domain, loader in self.val_loaders.items():
            correct = 0
            total = 0
            for im, label in loader:
                with torch.no_grad():
                    label = label.view(-1).cuda()
                    logits = self._predict_batch(im)
                    preds = logits.max(dim=1)[1]
                    correct += preds.eq(label).sum().item()
                    total += label.numel()
            accuracy = correct / total
            logging.info(f'    {self.name}.{domain}: {accuracy}')
            results[f'{self.name}.{domain}'] = accuracy
        self.net.train()
        return results


class GridHead(nn.Module):

    def __init__(self, ft_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rot = nn.Conv2d(ft_dim, 8, 1)

    def forward(self, x):
        x = self.pool(x)
        x = self.rot(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


class GridWrapper:

    def __init__(self, dataset, stride=256, transform=None, crop_transform=None):
        self.dataset = dataset
        self.stride = stride
        self.transform = transform
        self.crop_transform = crop_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im = self.dataset[index][0]
        if self.transform is not None:
            im = self.transform(im)
        crops = []
        for x in range(4):
            for y in range(2):
                crop = self.crop(im, x, y)
                if self.crop_transform is not None:
                    crop = self.crop_transform(crop)
                crops.append(crop)
        im = torch.stack(crops, dim=0)
        grid_label = list(range(8))
        return im, torch.LongTensor(grid_label)

    def crop(self, im, x, y):
        left = self.stride * x
        right = self.stride * (x + 1)
        up = self.stride * y
        down = self.stride * (y + 1)
        im = im.crop((left, up, right, down))
        return im
