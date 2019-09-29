import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from cityscapesscripts.helpers.labels import labels as cityscapes_labels


def remap_labels_to_train_ids(arr):
    arr = arr.copy()
    masks = []
    for label in cityscapes_labels:
        masks.append(arr == label.id)
    for label, mask in zip(cityscapes_labels, masks):
        arr[mask] = label.trainId
    # GTA has some pixels labeled as 34, we're just going to ignore them
    arr[arr == 34] = 255
    return arr


class Cityscapes(data.Dataset):

    num_classes = 19

    def __init__(self, root, split='train', remap_labels=True, transform=None,
                 target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform

    def collect_ids(self):
        im_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        ids = []
        for dirpath, dirnames, filenames in os.walk(im_dir):
            for filename in filenames:
                if filename.endswith('.png'):
                    ids.append('_'.join(filename.split('_')[:3]))
        return ids

    def img_path(self, id):
        fmt = 'leftImg8bit/{}/{}/{}_leftImg8bit.png'
        subdir = id.split('_')[0]
        path = fmt.format(self.split, subdir, id)
        return os.path.join(self.root, path)

    def label_path(self, id):
        fmt = 'gtFine/{}/{}/{}_gtFine_labelIds.png'
        subdir = id.split('_')[0]
        path = fmt.format(self.split, subdir, id)
        return os.path.join(self.root, path)

    def __getitem__(self, index):
        id = self.ids[index]
        img = Image.open(self.img_path(id)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = Image.open(self.label_path(id)).convert('L')
        if self.remap_labels:
            target = np.asarray(target)
            target = remap_labels_to_train_ids(target)
            target = Image.fromarray(target, 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)
