import os.path

import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from PIL import Image

from .cityscapes import remap_labels_to_train_ids


class GTA5(data.Dataset):

    num_classes = 19

    val_blacklist = [96, 173, 175, 334, 353]

    def __init__(self, root, split='train', remap_labels=True, transform=None,
                 target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        if split == 'val':
            self.ids = list(self.ids)
            for ind in self.val_blacklist[::-1]:
                del self.ids[ind]
        self.transform = transform
        self.target_transform = target_transform

    def collect_ids(self):
        splits = scipy.io.loadmat(os.path.join(self.root, 'split.mat'))
        ids = splits['{}Ids'.format(self.split)].squeeze()
        return ids

    def img_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.root, 'images', filename)

    def label_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.root, 'labels', filename)

    def __getitem__(self, index):
        id = self.ids[index]
        img_path = self.img_path(id)
        label_path = self.label_path(id)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = Image.open(label_path)
        if self.remap_labels:
            target = np.asarray(target)
            target = remap_labels_to_train_ids(target)
            target = Image.fromarray(target, 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)
