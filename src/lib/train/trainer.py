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
    def __init__(self, config):
        self.config = config

    def prepare_dataset(config):
        rdm = get_rdm(evc=int(config.evc))
        siamese_dataset = SiameseNetworkDataset(rdm=rdm,
                                                transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                              transforms.ToTensor()
                                                                              ]), should_invert=False, apply_foveate=config.foveate)

        train_dataloader = DataLoader(siamese_dataset,
                                      shuffle=True,
                                      num_workers=config.num_workers,
                                      batch_size=config.batch_size)
        return train_dataloader

    def train():

    def run_epoch():

    def log():
