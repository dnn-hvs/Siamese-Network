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
from utils.plot_images import imshow, show_plot
from utils.read_matfile import get_rdm
from loss.contrastive_loss import ContrastiveLoss
from loss.squared_euclidean_loss import EucledianLoss

from network.siamese_network import SiameseNetwork
from dataset.siamese_network_dataset import SiameseNetworkDataset


def prepare_dataset(config):
    folder_dataset = dset.ImageFolder(root=config.train_dir)
    rdm = get_rdm()
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, rdm=rdm,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ]), should_invert=False)

    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=config.batch_size)
    return train_dataloader


def train(train_dataloader, config):
    net = SiameseNetwork(config).cuda()
    print(net)
    criterion = EucledianLoss()
    # criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, config.num_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(
                    epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    show_plot(counter, loss_history)


def main(config):
    train_dataloader = prepare_dataset(config)
    train(train_dataloader, config)


if __name__ == '__main__':
    config = Config().parse()
    print(config)
    main(config)
