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
    def __init__(self, config, net, optimizer, logger):
        self.config = config
        self.net = net
        self.optimizer = optimizer
        self.logger = logger
        self.loss_criterion = EucledianLoss(logger)

    def set_device(self):
        # Multiple gpus support
        chunk_sizes = self.config.batch_size // len(self.config.gpus)
        if len(self.config.gpus) > 1:
            net = DataParallel(
                net, device_ids=self.config.gpus,
                chunk_sizes=chunk_sizes).to(self.config.device)
        else:
            net = self.net.to(self.config.device)
        return

    def train(self, epoch, train_dataloader):
        return self.run_epoch('train', epoch, train_dataloader)

    def run_epoch(self, phase, epoch, train_dataloader):
        total_iterations = len(train_dataloader)
        net = self.net
        if phase is 'train':
            net.train()
        else:
            if len(self.config.gpus) > 1:
                net = self.net.module
            net.eval()
            torch.cuda.empty_cache()
        pbar = tqdm(total=len(train_dataloader), desc='Batch', position=1)
        for i, data in enumerate(train_dataloader):
            img0, img1, label = data
            img0, img1, label = img0.to(device=self.config.device, non_blocking=True), img1.to(
                device=self.config.device, non_blocking=True), label.to(device=self.config.device, non_blocking=True)
            output1, output2 = net(img0, img1)
            loss = self.loss_criterion(output1, output2, label)
            if phase is 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.update(1)
            pbar.set_postfix(Loss=loss.item())
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
