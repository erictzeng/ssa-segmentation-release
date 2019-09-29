import numpy as np
import torch
from PIL import Image

from torchvision import transforms


class ContextDataset(object):

    def __init__(self, dataset, image_size=292, crop_size=255, jitter=21):
        self.dataset = dataset

        self.image_transform = transforms.Compose([
            #transforms.RandomGrayscale(0.66),
            transforms.Resize(image_size, Image.BILINEAR),
            transforms.RandomCrop(crop_size)
        ])
        tile_size = crop_size // 3
        self.tile_transform = transforms.Compose([
            transforms.RandomCrop(tile_size - jitter),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img, *_ = self.dataset[index]
        img = img.convert('RGB')
        img = self.image_transform(img)

        tiles = []
        stride = img.size[0] // 3
        for x in range(3):
            for y in range(3):
                tile = img.crop((x * stride, y * stride,
                                 (x + 1) * stride, (y + 1) * stride))
                tile = self.tile_transform(tile)
                #tile_mean = list(tile.view(3, -1).mean(1))
                #tile_std = list(tile.view(3, -1).std(1).max(torch.tensor(1 / np.sqrt(tile.numel()))))
                tile = transforms.functional.normalize(tile,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
                tiles.append(tile)
        tiles = [tiles[4]] + tiles[:4] + tiles[5:]
        tiles = torch.stack(tiles, 0)
        label = torch.LongTensor(range(8))
        return tiles, label

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def pair_batch(batch):
        ndim = batch.ndimension()
        if ndim == 4:
            n, c, h, w = batch.size()
            batch = batch.view(n // 9, 9, c, h, w)
        elif ndim == 2:
            n, c = batch.size()
            batch = batch.view(n // 9, 9, c)
        else:
            raise TypeError
        centers = batch[:, (0,)]
        edges = batch[:, 1:]
        paired = torch.cat([centers.expand_as(edges), edges], 2)
        if ndim == 4:
            paired = paired.view(-1, c * 2, h, w)
        elif ndim == 2:
            paired = paired.view(-1, c * 2)
        return paired
