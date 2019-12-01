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
alex_feat_list = ['conv1', 'ReLU1', 'maxpool1',
                  'conv2', 'ReLU2', 'maxpool2',
                  'conv3', 'ReLU3',
                  'conv4', 'ReLU4',
                  'conv5', 'ReLU5', 'maxpool5',
                  ]
alex_classifier_list = ['Dropout6', 'fc6',
                        'ReLU6', 'Dropout7', 'fc7', 'ReLU7', 'fc8']


class AlexNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(AlexNet, self).__init__()
        self.select_feats = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        self.select_classifier = ['fc6', 'fc7', 'fc8']

        self.feat_list = self.select_feats + self.select_classifier

        self.alex_feats = models.alexnet(pretrained=True).features
        self.alex_classifier = models.alexnet(pretrained=True).classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        """Extract multiple feature maps."""
        features = OrderedDict()
        count = 1
        for name, layer in self.alex_feats._modules.items():
            x = layer(x)
            features[str(count)+": "+layer.__class__.__name__] = x
            count += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        for name, layer in self.alex_classifier._modules.items():
            x = layer(x)
            features[str(count)+": "+layer.__class__.__name__] = x
            count += 1

        return features
