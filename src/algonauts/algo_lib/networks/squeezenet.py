from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
from collections import OrderedDict

sqnet1_0_feat_list = ['conv1', 'ReLU1', 'maxpool1',
                      'fire2',
                      'fire3',
                      'fire4', 'maxpool4',
                      'fire5',
                      'fire6',
                      'fire7',
                      'fire8', 'maxpool8',
                      'fire9',

                      ]
sqnet1_0_classifier_list = ['Dropout10', 'conv11', 'ReLU12', 'avgpool13']


sqnet1_1_feat_list = ['conv1', 'ReLU1', 'maxpool1',
                      'fire2',
                      'fire3', 'maxpool3',
                      'fire4',
                      'fire5', 'maxpool5',
                      'fire6',
                      'fire7',
                      'fire8',
                      'fire9',
                      ]
sqnet1_1_classifier_list = ['Dropout10', 'conv11', 'ReLU12', 'avgpool13']


class SqueezeNet1_0(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(SqueezeNet1_0, self).__init__()

        self.select_feats = ['maxpool1',
                             'maxpool4',
                             'fire8',
                             'maxpool8',
                             'fire9', ]
        self.select_classifier = ['conv11']

        self.feat_list = self.select_feats + self.select_classifier

        self.sqnet_feats = models.squeezenet1_0(pretrained=True).features
        self.sqnet_classifier = models.squeezenet1_0(
            pretrained=True).classifier

    def forward(self, x):
        """Extract multiple feature maps."""
        features = OrderedDict()
        count = 1
        for name, layer in self.sqnet_feats._modules.items():
            x = layer(x)
            features[str(count)+": "+layer.__class__.__name__] = (x)
            count += 1

        for name, layer in self.sqnet_classifier._modules.items():
            x = layer(x)
            features[str(count)+": "+layer.__class__.__name__] = (x)
            count += 1
        return features


class SqueezeNet1_1(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(SqueezeNet1_1, self).__init__()

        self.select_feats = ['maxpool1',
                             'maxpool3',
                             'maxpool5',
                             'fire7',
                             'fire9', ]
        self.select_classifier = ['conv11']

        self.feat_list = self.select_feats + self.select_classifier

        self.sqnet_feats = models.squeezenet1_1(pretrained=True).features
        self.sqnet_classifier = models.squeezenet1_1(
            pretrained=True).classifier

    def forward(self, x):
        """Extract multiple feature maps."""
        features = OrderedDict()
        count = 1
        for name, layer in self.sqnet_feats._modules.items():
            x = layer(x)
            features[str(count)+": "+layer.__class__.__name__] = (x)
            count += 1

        for name, layer in self.sqnet_classifier._modules.items():
            x = layer(x)
            features[str(count)+": "+layer.__class__.__name__] = (x)
            count += 1
        return features
