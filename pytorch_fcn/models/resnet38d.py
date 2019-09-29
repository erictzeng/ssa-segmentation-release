import torch
from torch import nn
import torchvision
import numpy as np

import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class Net(nn.Module):

    out_dim = 512

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])

    def __init__(self, num_classes=19, freeze=True, pretrained=False, double_conv=True):
        super(Net, self).__init__()
        self.freeze = freeze
        self.pretrained = pretrained
        self.double_conv = double_conv

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)

        self.bn7 = nn.BatchNorm2d(4096)
        if double_conv:
            self.conv6a = nn.Conv2d(4096, 512, kernel_size=3, dilation=12, padding=12, bias=True)
            self.linear19 = nn.Conv2d(512, 19, kernel_size=3, dilation=12, padding=12, bias=True)
        else:
            self.fc = nn.Conv2d(4096, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)

        #self.not_training = [self.conv1a]
        #self.not_training = [
        #        self.conv1a,
        #        self.b2,
        #        self.b2_1,
        #        self.b2_2,
        #        self.b3,
        #        self.b3_1,
        #        self.b3_2,
        #        self.b4,
        #        self.b4_1,
        #        self.b4_2,
        #        self.b4_3,
        #        self.b4_4,
        #        self.b4_5,
        #]
        self.not_training = []


        self.normalize = Normalize()

        return

    def forward(self, x):
        y = x = self.forward_as_dict(x)['conv6']
        if self.double_conv:
            n, c, h, w = x.size()
            y = x = F.relu(self.conv6a(x))
            x = self.linear19(x)
        else:
            x = self.fc(x)
        return x, [y]


    def forward_as_dict(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))

        return dict({'conv4': conv4, 'conv5': conv5, 'conv6': conv6})


    def train(self, mode=True):

        super().train(mode)

        if self.freeze:
            for layer in self.not_training:

                if isinstance(layer, torch.nn.Conv2d):
                    layer.weight.requires_grad = False

                elif isinstance(layer, torch.nn.Module):
                    for c in layer.children():
                        c.weight.requires_grad = False
                        if c.bias is not None:
                            c.bias.requires_grad = False

            for layer in self.modules():

                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.eval()
                    #layer.bias.requires_grad = True
                    #layer.weight.requires_grad = True


def convert_mxnet_to_torch(filename):
    import mxnet

    save_dict = mxnet.nd.load(filename)

    renamed_dict = dict()

    bn_param_mx_pt = {'beta': 'bias', 'gamma': 'weight', 'mean': 'running_mean', 'var': 'running_var'}

    for k, v in save_dict.items():

        v = torch.from_numpy(v.asnumpy())
        toks = k.split('_')

        if 'conv1a' in toks[0]:
            renamed_dict['conv1a.weight'] = v

        elif 'linear1000' in toks[0]:
            pass

        elif 'branch' in toks[1]:

            pt_name = []

            if toks[0][-1] != 'a':
                pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
            else:
                pt_name.append('b' + toks[0][-2])

            if 'res' in toks[0]:
                layer_type = 'conv'
                last_name = 'weight'

            else:  # 'bn' in toks[0]:
                layer_type = 'bn'
                last_name = bn_param_mx_pt[toks[-1]]

            pt_name.append(layer_type + '_' + toks[1])

            pt_name.append(last_name)

            torch_name = '.'.join(pt_name)
            renamed_dict[torch_name] = v

        else:
            last_name = bn_param_mx_pt[toks[-1]]
            renamed_dict['bn7.' + last_name] = v

    return renamed_dict

