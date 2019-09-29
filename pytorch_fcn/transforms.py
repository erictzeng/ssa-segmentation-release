import numbers
import random

import numpy as np
import torch
import torchvision


def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))


class RandomCrop(object):
    """Crops the given tensors at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, tensors):
        output = []
        h, w = None, None
        th, tw = self.size
        for tensor in tensors:
            if h is None and w is None:
                _, h, w = tensor.size()
            elif tensor.size()[-2:] != (h, w):
                print(tensor.size(), (h, w))
                raise ValueError('Images must be same size')
        if w == tw and h == th:
            return tensors
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for tensor in tensors:
            output.append(tensor[..., y1:y1 + th, x1:x1 + tw].contiguous())
        return output


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given tensors with a probability of 0.5
    """

    def __call__(self, tensors):
        if random.random() < 0.5:
            output = []
            for tensor in tensors:
                indices = torch.arange(tensor.size(-1) - 1, -1, -1).long()
                output.append(tensor.index_select(-1, indices))
            return output
        return tensors


class RandomRotation(object):

    def __call__(self, tensors):
        direction = random.randrange(4)
        if direction == 1:
            tensors = [tensor.transpose(-2, -1).flip(-2) for tensor in tensors]
        elif direction == 2:
            tensors = [tensor.flip(-2).flip(-1) for tensor in tensors]
        elif direction == 3:
            tensors = [tensor.flip(-2).transpose(-2, -1) for tensor in tensors]
        return tensors

class AugmentCollate(object):

    def __init__(self, crop=None, flip=False, rotate=False):
        self.crop = crop
        self.flip = flip
        transforms = []
        if crop is not None:
            transforms.append(RandomCrop(crop))
        if rotate:
            transforms.append(RandomRotation())
        if flip:
            transforms.append(RandomHorizontalFlip())
        self.transform = torchvision.transforms.Compose(transforms)

    def __call__(self, batch):
        batch = [self.transform(x) for x in batch]
        return torch.utils.data.dataloader.default_collate(batch)
