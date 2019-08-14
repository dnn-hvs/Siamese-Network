import torch
# import torch.nn as nn
import torch.nn.functional as F


class EucledianLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(EucledianLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        predicted = 1 - self.pearson_corelation(output1, output2)
        # print(predicted, predicted.shape, label.shape)
        euclidean_distance = F.pairwise_distance(
            predicted, label, keepdim=True)
        # print(euclidean_distance.size())
        return torch.mean(euclidean_distance)

    def pearson_corelation(self, output1, output2):
        x = output1
        y = output2
        vx = x - torch.mean(x, axis=1).unsqueeze(1)
        vy = y - torch.mean(y, axis=1).unsqueeze(1)

        # print(torch.sum(vx * vy, axis=1).shape)
        return (torch.sum(vx * vy, axis=1) / (torch.sqrt(torch.sum(vx ** 2, axis=1)) * torch.sqrt(torch.sum(vy ** 2, axis=1)))).unsqueeze(1)
