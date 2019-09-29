import click
import numpy as np
import torch
from torch.utils.data import ConcatDataset

import pytorch_fcn.models.resnet38d as resnet38d
from pytorch_fcn.data import datasets
from pytorch_fcn.data import get_dataset
from pytorch_fcn.data.wrappers import RandomRescaleWrapper
from pytorch_fcn.models.drn import drn_c_26
from pytorch_fcn.models.deeplab import resnet101 as deeplab_resnet101
from pytorch_fcn.models.task_net import TaskNet
from pytorch_fcn.tasks import AdversarialAdaptation
from pytorch_fcn.tasks import ContinuousGridRegression
from pytorch_fcn.tasks import GridRegression
from pytorch_fcn.tasks import Rotation
from pytorch_fcn.tasks import Segmentation
from pytorch_fcn.tasks import VerticalFlip
from pytorch_fcn.trainer import TaskTrainer
from pytorch_fcn.util import config_logging


class RandomSubset:
    def __init__(self, dataset):
        self.dataset = dataset
        random = np.random.RandomState(seed=12345)
        self.perm = random.permutation(len(dataset))[:500]

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, index):
        return self.dataset[self.perm[index]]


@click.command()
@click.argument('output')
@click.option('--model', '-m', default='drn')
@click.option('--source', '-s', default='gta5')
@click.option('--lr', '-l', default=0.0016)
@click.option('--batch-size-per-gpu', '-b', default=8)
def main(output, model, source, lr, batch_size_per_gpu):
    config_logging()
    num_gpus = torch.cuda.device_count()
    batch_size = batch_size_per_gpu * num_gpus
    iterations = 16000
    step = 16000


    # datasets
    if '+' in source:
        sources = source.split('+')
        datasets = [get_dataset(source, split='all') for source in sources]
        source_dataset = ConcatDataset(datasets)
        source_dataset.num_classes = 19
        val_datasets = [get_dataset(source, split='val') for source in sources]
        source_val_dataset = ConcatDataset(val_datasets)
        source_val_dataset.num_classes = 19
    else:
        source_dataset = get_dataset(source, split='all')
        source_val_dataset = get_dataset(source, split='val')

    source_dataset = RandomRescaleWrapper(source_dataset, min_scale=0.5, max_scale=2.0, w=2048, h=1024)
    target_dataset = RandomRescaleWrapper(get_dataset('cityscapes'), min_scale=0.5, max_scale=2.0)
    source_val_dataset = RandomSubset(source_val_dataset)
    target_val_dataset = get_dataset('cityscapes', split='val')

    # net
    num_classes = source_dataset.num_classes
    if model == 'drn':
        backbone = drn_c_26(
            pretrained=True,
            finetune=True,
            num_classes=num_classes,
            out_map=True,
            out_middle=True
        )
    elif model == 'r38':
        backbone = resnet38d.Net(
            num_classes=num_classes,
            freeze=False,
            pretrained=True,
        )
        state_dict = backbone.state_dict()
        state_dict.update(torch.load('gta_rna-a1_cls19_s8_ep-0000.pth'))
        backbone.load_state_dict(state_dict)
        backbone.freeze = True
        backbone.train()
    elif model == 'deeplab':
        backbone = deeplab_resnet101(
            num_classes=num_classes,
            pretrained=True,
            freeze=True,
        )
        backbone.load_state_dict(torch.load('results/deeplab-gta5-sourceonly/snapshot/net-iter016000.pth'))
        backbone.train()
    else:
        raise KeyError(model)
    net = TaskNet(backbone)

    # tasks
    tasks = [
        Segmentation(net, source_dataset, source_val_dataset, target_val_dataset, batch_size=batch_size, crop_size=512),
        Rotation(net, source_dataset, target_dataset, source_val_dataset, target_val_dataset,
                 batch_size=batch_size // 2, crop_size=100, name='rotation100'),
        Rotation(net, source_dataset, target_dataset, source_val_dataset, target_val_dataset,
                 batch_size=batch_size // 2, crop_size=200, name='rotation200'),
        Rotation(net, source_dataset, target_dataset, source_val_dataset, target_val_dataset,
                 batch_size=batch_size // 2, crop_size=400, name='rotation400'),
        ContinuousGridRegression(net, source_dataset, target_dataset, source_val_dataset, target_val_dataset,
                                 batch_size=batch_size // 2, crop_size=100, name='gridregression100'),
        ContinuousGridRegression(net, source_dataset, target_dataset, source_val_dataset, target_val_dataset,
                                 batch_size=batch_size // 2, crop_size=200, name='gridregression200'),
        ContinuousGridRegression(net, source_dataset, target_dataset, source_val_dataset, target_val_dataset,
                                 batch_size=batch_size // 2, crop_size=400, name='gridregression400'),
    ]

    net = torch.nn.DataParallel(net).cuda()
    for task in tasks:
        task.net = net
    trainer = TaskTrainer(output, net, tasks, iterations=iterations, lr=lr, momentum=0.9, step_lr=step)
    trainer.run()


if __name__ == '__main__':
    main()
