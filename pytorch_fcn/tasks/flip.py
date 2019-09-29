import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_fcn.data.util import Dispenser
from pytorch_fcn.data.util import JointDispenser

class VerticalFlip:

    def __init__(self, net, source_dataset, target_dataset, source_val_dataset, target_val_dataset, *, batch_size, crop_size=600, name='vertflip'):
        self.net = net
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_val_dataset = source_val_dataset
        self.target_val_dataset = target_val_dataset
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.name = name

        self.loss_fn = nn.NLLLoss()
        self.create_datasets()
        self.create_head()

    def create_datasets(self):
        transform = transforms.Compose([
            transforms.Resize(1024),
            transforms.RandomCrop(self.crop_size),
            self.net.transform,
        ])
        loaders = []
        for dataset in [self.source_dataset, self.target_dataset]:
            flip_dataset = FlipWrapper(
                dataset,
                transform=transform,
            )
            loader = DataLoader(
                flip_dataset,
                batch_size=self.batch_size // 2,
                shuffle=True,
                num_workers=4
            )
            loaders.append(loader)
        val_loaders = []
        for dataset in [self.source_val_dataset, self.target_val_dataset]:
            flip_dataset = FlipWrapper(
                dataset,
                transform=transform,
            )
            loader = DataLoader(
                flip_dataset,
                batch_size=2,
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
        self.head = FlipHead(self.net.out_dim)
        self.net.attach_head(self.name, self.head)

    def _predict_batch(self, im):
        n, g, c, h, w = im.size()
        im = im.view(n * g, c, h, w).cuda()
        preds = self.net(im, task=self.name)
        preds = preds.view(n * g, 2)
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


class FlipHead(nn.Module):

    def __init__(self, ft_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flip = nn.Conv2d(ft_dim, 2, 1)

    def forward(self, x):
        x = self.pool(x)
        x = self.flip(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


class FlipWrapper:

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im = self.dataset[index][0]
        flipped = im.transpose(Image.FLIP_TOP_BOTTOM)
        flips = [im, flipped]
        if self.transform is not None:
            flips = [self.transform(flip) for flip in flips]
        im = torch.stack(flips, dim=0)
        flip_label = list(range(2))
        return im, torch.LongTensor(flip_label)
