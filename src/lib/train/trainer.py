import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm, trange
# import _init_paths
import os
from datetime import datetime
from utils.config import Config
from utils.plot_images import imshow, save_plot
from utils.read_matfile import get_rdm
from loss.contrastive_loss import ContrastiveLoss
from loss.squared_euclidean_loss import EucledianLoss
from train.model_utils import save_model, load_model
from network.siamese_network import SiameseNetwork
from dataset.siamese_network_dataset import SiameseNetworkDataset
from train.data_parallel import DataParallel


class Trainer():
    def __init__(self, config, net, optimizer):
        self.config = config
        self.net = net
        self.optimizer = optimizer
        self.loss_criterion = EucledianLoss()

    def set_device(self, config, device):
        # Multiple gpus support
        chunk_sizes = config.batch_size // len(config.gpus)
        if len(config.gpus) > 1:
            net = DataParallel(
                net, device_ids=config.gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            net = net.to(device)
        return

    def train(epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    def run_epoch(self, phase, epoch, data_loader):
        total_iterations = len(train_dataloader)
        net = self.net
        if phase is 'train':
            net.train()
        else:
            if len(self.config.gpus) > 1:
                net = self.net.module
            net.eval()
            torch.cuda.empty_cache()

        for i, data in enumerate(train_dataloader):
            img0, img1, label = data
            img0, img1, label = img0.to(device=device, non_blocking=True), img1.to(
                device=device, non_blocking=True), label.to(device=device, non_blocking=True)
            output1, output2 = net(img0, img1)
            loss = self.loss_criterion(output1, output2, label)
            if phase is 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return loss.item()

    def freeze(self):
        ct = 0
        for name, child in self.net.named_children():
            for name2, params in self.net.named_parameters():
                if self.config.gt:
                    if ct > self.config.num_freeze_layers*2:
                        print("Freezing layer:", name2)
                        params.requires_grad = False
                else:
                    if ct < self.config.num_freeze_layers*2:
                        print("Freezing layer:", name2)
                        params.requires_grad = False
                ct += 1
        return
