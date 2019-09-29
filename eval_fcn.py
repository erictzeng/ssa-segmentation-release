import os

import click
import numpy as np
import torch
import torchvision

from pytorch_fcn.data import datasets
from pytorch_fcn.data import get_dataset
from pytorch_fcn.models.drn import drn_c_26
from pytorch_fcn.transforms import to_tensor_raw


def fmt_array(arr):
    strs = ['{:.3f}'.format(x) for x in arr]
    return '  '.join(strs)


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--dataset', default='cityscapes',
              type=click.Choice(datasets.keys()))
def main(path, dataset):
    net = drn_c_26(num_classes=19, out_map=True)
    net.eval()
    net.cuda()
    ds = get_dataset(dataset, split='val', transform=net.transform,
                     target_transform=to_tensor_raw)
    loader = torch.utils.data.DataLoader(ds, num_workers=8)
    net.load_state_dict(torch.load(path))

    intersections = np.zeros(19)
    unions = np.zeros(19)

    errs = []
    print(len(loader))
    for im_i, (im, label) in enumerate(loader):
        im = im.cuda()
        label = label.cuda()
        _, h, w = label.size()
        with torch.no_grad():
            logits = net(im)
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
        _, preds = torch.max(logits, 1)
        try:
            preds[torch.eq(label, 255)] = 255
            for i in range(19):
                pred_mask = torch.eq(preds, i)
                gt_mask = torch.eq(label, i)
                intersections[i] += torch.sum(pred_mask & gt_mask)
                unions[i] += torch.sum(pred_mask | gt_mask)
        except (RuntimeError, IndexError):
            errs.append(im_i)
        so_far = np.maximum(intersections, 1) / np.maximum(unions, 1)
        print('im {}'.format(im_i))
        print(fmt_array(so_far))
        print(np.mean(so_far))
        print()
    print(fmt_array(so_far))
    print(np.mean(so_far))
    print()
    print(errs)

if __name__ == '__main__':
    main()
