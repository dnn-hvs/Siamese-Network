import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
from torch import optim

import _init_paths

from utils.config import Config
from utils.plot_images import imshow, save_plot
from utils.read_matfile import get_rdm
from loss.contrastive_loss import ContrastiveLoss
from loss.squared_euclidean_loss import EucledianLoss

from network.siamese_network import SiameseNetwork
from dataset.siamese_network_dataset import SiameseNetworkDataset


def prepare_dataset(config):
    folder_dataset = dset.ImageFolder(root=config.train_dir)
    print(type(folder_dataset))
    rdm = get_rdm()
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, rdm=rdm,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ]), should_invert=False, apply_foveate=config.foveate)

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  batch_size=config.batch_size)
    return train_dataloader


def train(train_dataloader, config):
    net = SiameseNetwork(config).cuda()
    loss_criterion = EucledianLoss()
    # loss_criterion = ContrastiveLoss()

    # print(net.named_children())
    # Freezing the first few layers. Here I am freezing the first 7 layers ct = 0
    ct = 0
    for name, child in net.named_children():
        for name2, params in net.named_parameters():
            if config.gt:
                if ct > config.num_freeze_layers*2:
                    print("Freezing layer:", name2)
                    params.requires_grad = False
            else:
                if ct < config.num_freeze_layers*2:
                    print("Freezing layer:", name2)
                    params.requires_grad = False
            ct += 1

    # print(net.parameters())
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0
    for epoch in range(0, config.num_epochs):
        num_images = 0

        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            # print(img0.shape)
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            num_images += img0.shape[0]
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss = loss_criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}; Current loss {};".format(
                    epoch, loss.item()), end='\r')
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())

        print("\nTotal Number of images: ", num_images)

    save_plot(counter, loss_history, config.plot_name)


def main(config):
    train_dataloader = prepare_dataset(config)
    train(train_dataloader, config)


if __name__ == '__main__':
    config = Config().parse()
    print(config)
    main(config)
