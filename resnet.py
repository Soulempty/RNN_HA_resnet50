from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

__all__=['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }
    
    def __init__(self,depth,classes,pretrained=True):
        super(ResNet,self).__init__()
        self.num_class=classes
        self.depth=depth
        self.pretrained=pretrained
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base=ResNet.__factory[depth](pretrained=pretrained)
        out_planes = self.base.fc.in_features
        self.base.fc=nn.Linear(out_planes, self.num_class)
        init.normal(self.base.fc.weight,std=0.001)
        init.constant(self.base.fc.bias, 0)
    def forward(self,x):
        feature=self.base(x)
        return feature

def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)
