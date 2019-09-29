import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_fcn.data.util import Dispenser
from pytorch_fcn.data.util import JointDispenser
from pytorch_fcn.data.wrappers import TransformWrapper
from pytorch_fcn.transforms import AugmentCollate
from pytorch_fcn.transforms import to_tensor_raw


class AdversarialAdaptation:

    def __init__(self, net, source_dataset, target_dataset, source_val_dataset, target_val_dataset, *, batch_size, crop_size, name='adversarial_adapt'):
        self.name = name
        self.net = net
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_val_dataset = source_val_dataset
        self.target_val_dataset = target_val_dataset

        self.adv_opt = torch.optim

        self.batch_size = batch_size
        self.crop_size = crop_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.create_datasets()
        self.create_head()

    def create_datasets(self):
        transform = transforms.Compose([
            self.net.transform,
        ])
        target_transform = transforms.Compose([
            to_tensor_raw,
        ])
        collate_fn = AugmentCollate(crop=self.crop_size, flip=True)
        train_loaders = []
        for train_dataset in [self.source_dataset, self.target_dataset]:
            dataset = TransformWrapper(
                train_dataset,
                transform=transform,
                target_transform=target_transform
            )
            train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size // 2,
                shuffle=True,
                num_workers=8,
                collate_fn=collate_fn,
                drop_last=True
            )
            train_loaders.append(train_loader)
        self.train_dispenser = JointDispenser(*train_loaders)

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
        self.adv_dis = AdversarialDiscriminator(self.net.out_dim).cuda()
        self.adv_opt = torch.optim.SGD(self.adv_dis.parameters(), lr=0.007, momentum=0.9)

    def _predict_batch(self, im):
        im = im.cuda()
        preds = self.net(im, task=self.name)
        return preds

    def step(self):
        self.step_discriminator()
        return self.step_generator()

    def step_discriminator(self):
        im, _ = self.train_dispenser.next_batch()
        im = im.cuda()
        label = self.generate_label(im)
        self.adv_opt.zero_grad()
        with torch.no_grad():
            _, fts = self.net(im, task=None)
        preds = self.adv_dis(fts)
        loss = self.loss_fn(preds, label)
        loss.backward()
        self.adv_opt.step()

    def step_generator(self):
        im, _ = self.train_dispenser.next_batch()
        im = im.cuda()
        label = self.generate_label(im, invert=True)
        _, fts = self.net(im, task=None)
        preds = self.adv_dis(fts)
        loss = self.loss_fn(preds, label)
        return loss

    def generate_label(self, im, invert=False):
        zeros = torch.zeros(im.size(0) // 2, device=im.device, dtype=torch.long)
        ones = torch.ones(im.size(0) // 2, device=im.device, dtype=torch.long)
        if invert:
            return torch.cat([ones, zeros], dim=0)
        return torch.cat([zeros, ones], dim=0)

    def eval(self):
        return {}


class AdversarialDiscriminator(nn.Module):

    def __init__(self, ft_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ft_dim, 2, 1)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
