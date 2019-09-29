import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from pytorch_fcn.data.util import Dispenser
from pytorch_fcn.data.wrappers import RandomRescaleWrapper
from pytorch_fcn.data.wrappers import TransformWrapper
from pytorch_fcn.transforms import AugmentCollate
from pytorch_fcn.transforms import to_tensor_raw


class Segmentation:
    """NOTE: segmentation is a special case, only needs source"""

    def __init__(self, net, source_dataset, source_val_dataset, target_val_dataset, name='segmentation', batch_size=8, crop_size=400):
        self.net = net
        self.dataset = source_dataset
        self.source_val_dataset = source_val_dataset
        self.target_val_dataset = target_val_dataset
        self.name = name
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.loss_fn = nn.NLLLoss(ignore_index=255, reduction='sum')
        self.create_datasets()
        self.create_head()

    def create_datasets(self):
        transform = transforms.Compose([
            self.net.transform,
        ])
        target_transform = transforms.Compose([
            to_tensor_raw,
        ])
        collate_fn = AugmentCollate(crop=self.crop_size, flip=True, rotate=False)
        dataset = TransformWrapper(
            self.dataset,
            transform=transform,
            target_transform=target_transform
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn
        )
        self.train_dispenser = Dispenser(loader)
        val_transform = transforms.Compose([
            transforms.Resize((1024, 2048)),
            self.net.transform,
        ])
        val_target_transform = transforms.Compose([
            transforms.Resize((1024, 2048), interpolation=Image.NEAREST),
            to_tensor_raw,
        ])
        val_loaders = []
        for dataset in [self.source_val_dataset, self.target_val_dataset]:
            dataset = TransformWrapper(
                dataset,
                transform=val_transform,
                target_transform=val_target_transform,
            )
            loader = DataLoader(
                dataset,
                batch_size=torch.cuda.device_count(),
                num_workers=4,
            )
            val_loaders.append(loader)
        self.val_loaders = {
            'source': val_loaders[0],
            'target': val_loaders[1],
        }

    def create_head(self):
        # no head needed, use the model itself
        pass

    def _predict_batch(self, im):
        _, _, h, w = im.size()
        im = im.cuda()
        preds, ft = self.net(im)
        preds = nn.functional.log_softmax(preds, dim=1)
        return preds, ft

    def step(self):
        im, label = self.train_dispenser.next_batch()
        label = label.cuda()
        log_probs, _ = self._predict_batch(im)
        _, _, h, w = log_probs.size()
        loss = self.loss_fn(log_probs, label)
        denom = (label != 255).sum()
        loss /= denom
        num_gpus = torch.cuda.device_count()
        loss /= num_gpus
        return loss

    def eval(self):
        self.net.eval()
        results = {}
        mean_fts = {}
        for domain, loader in self.val_loaders.items():
            intersections = np.zeros(self.dataset.num_classes)
            unions = np.zeros(self.dataset.num_classes)
            skipped = 0
            for i, (im, label) in enumerate(loader):
                _, _, h, w = im.size()
                _, lh, lw = label.size()
                if h != lh or w != lw:
                    skipped += 1
                    logging.warn(f'Image {i} had size mismatch')
                    continue
                label = label.cuda()
                with torch.no_grad():
                    log_probs, ft = self._predict_batch(im)
                    if log_probs.size(2) != lh or log_probs.size(3) != lw:
                        log_probs = nn.functional.interpolate(log_probs, size=(h, w), mode='bilinear', align_corners=True)
                n, c, h, w = ft.size()
                ft = ft.view(n, c, h * w).mean(2)
                if domain not in mean_fts:
                    mean_fts[domain] = []
                mean_fts[domain].append(ft)
                preds = log_probs.max(dim=1)[1]
                preds[label == 255] = 255
                for class_i in range(self.dataset.num_classes):
                    pred_mask = (preds == class_i)
                    gt_mask = (label == class_i)
                    intersections[class_i] += (pred_mask & gt_mask).sum()
                    unions[class_i] += (pred_mask | gt_mask).sum()
            ious = np.maximum(intersections, 1) / np.maximum(unions, 1)
            miou = np.mean(ious)
            logging.info(f'    {self.name}.{domain}: {miou}')
            results[f'{self.name}.{domain}'] = ious
        src_mean = torch.cat(mean_fts['source'], 0).mean(0)
        tgt_mean = torch.cat(mean_fts['target'], 0).mean(0)
        mmd = torch.dist(src_mean, tgt_mean, 2).item()
        logging.info(f'    mmd: {mmd}')
        results['mmd'] = mmd
        self.net.train()
        return results
