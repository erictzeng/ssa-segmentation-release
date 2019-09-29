import math
import random

import numpy as np
import torch
from PIL import Image

import pytorch_fcn.color.color as color

class RandomRescaleWrapper:

    def __init__(self, dataset, min_scale=0.7, max_scale=1.3, w=None, h=None):
        self.dataset = dataset
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.width = w
        self.height = h

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im, target = self.dataset[index]
        if self.width is None or self.height is None:
            w, h = im.size
        else:
            w, h = self.width, self.height
        log_min = math.log(self.min_scale)
        log_max = math.log(self.max_scale)
        scale = math.exp(random.uniform(log_min, log_max))
        #scale = random.uniform(self.min_scale, self.max_scale)
        new_w = round(w * scale)
        new_h = round(h * scale)
        im = im.resize((new_w, new_h), Image.LANCZOS)
        target = target.resize((new_w, new_h), Image.NEAREST)
        return im, target

    @property
    def num_classes(self):
        return self.dataset.num_classes


class TransformWrapper:
    """Utility class for applying transforms to a raw dataset."""

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im, target = self.dataset[index]
        if self.transform is not None:
            im = self.transform(im)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return im, target


class RotationWrapper:

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im = self.dataset[index][0]
        if self.transform is not None:
            im = self.transform(im)
        rotation_label = list(range(4))
        im = torch.stack([self.flip(im, i) for i in range(4)], 0)
        return im, torch.LongTensor(rotation_label)

    def flip(self, im, direction):
        if direction == 1:
            im = im.transpose(1, 2).flip(1)
        elif direction == 2:
            im = im.flip(1).flip(2)
        elif direction == 3:
            im = im.flip(1).transpose(1, 2)
        return im

    def random_flip(self, im):
        label = random.randrange(4)
        return self.flip(im, label), label


class ColorWrapper:

    def __init__(self, dataset, normalize=None):
        self.dataset = dataset
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im = self.dataset[index][0]
        color_data = color.get_colorization_data(im.unsqueeze(0))
        color_label = color.encode_ab_ind(color_data['B']).long()[0]
        color_label = color_label.squeeze(0)
        im = im.mean(0, keepdim=True).expand(3, -1, -1)
        if self.normalize is not None:
            im = self.normalize(im)
        return im, color_label


class ColorRegressionWrapper:

    def __init__(self, dataset, normalize=None, downscale_factor=8.):
        self.dataset = dataset
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im = self.dataset[index][0]
        color_data = color.get_colorization_data(im.unsqueeze(0))
        color_data = color_data['B'].squeeze(0)
        im = im.mean(0, keepdim=True).expand(3, -1, -1)
        if self.normalize is not None:
            im = self.normalize(im)
        return im, color_data


class RandomSubset:
    def __init__(self, dataset):
        self.dataset = dataset
        random = np.random.RandomState(seed=12345)
        self.perm = random.permutation(len(dataset))[:500]

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, index):
        return self.dataset[self.perm[index]]
