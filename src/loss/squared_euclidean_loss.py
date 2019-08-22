import torch
# import torch.nn as nn
import torch.nn.functional as F
from utils.logger import Logger


class EucledianLoss(torch.nn.Module):
    def __init__(self, logger):
        super(EucledianLoss, self).__init__()
        self.logger = logger

    def forward(self, output1, output2, label):
        last_layer_index = len(output1) - 1
        predicted = 1 - \
            self.pearson_corelation(
                output1[last_layer_index], output2[last_layer_index])
        # print(predicted, predicted.shape, label.shape)
        euclidean_distance = F.pairwise_distance(
            predicted, label, keepdim=True) ** 2

        if not torch.mean(euclidean_distance):
            exit()

        return torch.mean(euclidean_distance)

    def pearson_corelation(self, x, y):
        x_mean = torch.mean(x, axis=1)
        y_mean = torch.mean(y, axis=1)
        vx = x - x_mean.unsqueeze(1)
        vy = y - y_mean.unsqueeze(1)

        numerator = torch.sum(vx * vy, axis=1)
        denominator1 = torch.sqrt(torch.sum(vx ** 2, axis=1))
        denominator2 = torch.sqrt(torch.sum(vy ** 2, axis=1))
        denominator = denominator1 * denominator2

        log_txt = f"Mean x:  {x_mean} \nMean y: {y_mean} \nNumerator: {numerator} \
        \nDenominator: {denominator}\n"
        log_txt += "="*100 + "\n"

        self.logger.write(txt=log_txt)
        return (numerator / denominator+1e-5).unsqueeze(1)
