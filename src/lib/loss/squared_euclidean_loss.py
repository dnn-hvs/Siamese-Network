import torch
# import torch.nn as nn
import torch.nn.functional as F


class EucledianLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(EucledianLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        last_layer_index = len(output1) - 1
        predicted = 1 - \
            self.pearson_corelation(
                output1[last_layer_index], output2[last_layer_index])
        # print(predicted, predicted.shape, label.shape)
        euclidean_distance = F.pairwise_distance(
            predicted, label, keepdim=True) ** 2
        return torch.mean(euclidean_distance)

    def pearson_corelation(self, x, y):
        vx = x - torch.mean(x, axis=1).unsqueeze(1)
        vy = y - torch.mean(y, axis=1).unsqueeze(1)

        return (torch.sum(vx * vy, axis=1) / (torch.sqrt(torch.sum(vx ** 2, axis=1)) * torch.sqrt(torch.sum(vy ** 2, axis=1)))).unsqueeze(1)
