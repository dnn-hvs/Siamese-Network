import torch.nn as nn

import torch
from vgg import *
from resnet import *
from alexnet import *
from squeezenet import *
from densenet import *
from inceptionv3 import *
from googlenet import *

models = {
    'alexnet': AlexNet,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'sqnet1_0': SqueezeNet1_0,
    'sqnet1_1': SqueezeNet1_1,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'googlenet': googlenet,
    'inception': inception_v3
}


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        if config.arch == 'alexnet':
            self.network = models[config.arch]()
        else:
            self.network = models[config.arch](pretrained=True)

    def forward(self, input1, input2):
        output1 = self.network(input1)
        output2 = self.network(input2)
        return output1, output2
