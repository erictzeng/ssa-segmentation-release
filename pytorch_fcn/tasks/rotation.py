import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_fcn.data.util import Dispenser
from pytorch_fcn.data.util import JointDispenser
from pytorch_fcn.data.wrappers import RotationWrapper


class Rotation:

    def __init__(self, net, source_dataset, target_dataset, source_val_dataset, target_val_dataset, *, batch_size, name='rotation', crop_size=400):
        self.net = net
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_val_dataset = source_val_dataset
        self.target_val_dataset = target_val_dataset
        self.batch_size = batch_size
        self.name = name
        self.crop_size = crop_size

        self.loss_fn = nn.NLLLoss()
        self.create_datasets()
        self.create_head()

    def create_datasets(self):
        transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            self.net.transform,
        ])
        loaders = []
        for dataset in [self.source_dataset, self.target_dataset]:
            rot_dataset = RotationWrapper(dataset, transform=transform)
            loader = DataLoader(
                rot_dataset,
                batch_size=self.batch_size // 2,
                shuffle=True,
                num_workers=4
            )
            loaders.append(loader)
        self.train_dispenser = JointDispenser(*loaders)
        val_loaders = []
        for dataset in [self.source_val_dataset, self.target_val_dataset]:
            rot_dataset = RotationWrapper(dataset, transform=transform)
            loader = DataLoader(
                rot_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=4
            )
            val_loaders.append(loader)
        self.val_loaders = {
            'source': val_loaders[0],
            'target': val_loaders[1],
        }

    def create_head(self):
        self.head = RotationHead(self.net.out_dim)
        self.net.attach_head(self.name, self.head)

    def _predict_batch(self, im):
        n, r, c, h, w = im.size()
        im = im.view(n * r, c, h, w).cuda()
        preds = self.net(im, task=self.name)
        preds = preds.view(n * r, 4)
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


class RotationHead(nn.Module):

    def __init__(self, ft_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rot = nn.Conv2d(ft_dim, 4, 1)

    def forward(self, x):
        x = self.pool(x)
        x = self.rot(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
