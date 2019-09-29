import torch
import torch.nn as nn


class TaskNet(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.heads = nn.ModuleDict()

    def attach_head(self, name, head):
        if name in self.heads:
            raise KeyError(f'duplicate head {name}')
        self.heads[name] = head

    def forward(self, x, task=None, backbone_grad=True, head_grad=True):
        if not backbone_grad:
            with torch.no_grad():
                x, fts = self.net(x)
        else:
            x, fts = self.net(x)
        if task is not None:
            x = fts[-1]
            if not head_grad:
                with torch.no_grad():
                    x = self.heads[task](x)
            else:
                x = self.heads[task](x)
            return x
        else:
            return x, fts[-1]

    @property
    def transform(self):
        return self.net.transform

    @property
    def out_dim(self):
        return self.net.out_dim

    def freeze_batch_norm(self):
        for module in self.modules():
            if isinstance(module, nn.modules.BatchNorm2d):
                module.eval()
