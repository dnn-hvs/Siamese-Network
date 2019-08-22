import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm, trange
import _init_paths
import os
import sys
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
from utils.logger import Logger


def prepare_dataset(config):
    rdm = get_rdm(evc=int(config.evc))
    siamese_dataset = SiameseNetworkDataset(rdm=rdm,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                                                               0.229, 0.224, 0.225])
                                                                          ]), should_invert=False, apply_foveate=config.foveate)

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  batch_size=config.batch_size)
    return train_dataloader


def train(train_dataloader, config, logger):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus_str
    device = torch.device('cuda' if config.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    net = SiameseNetwork(config)
    net.to(device)

    if config.optim == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr)
    elif config.optim == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     net.parameters()), lr=config.lr, momentum=0.78)

    loss_criterion = EucledianLoss()

    # Resume the model if needed
    start_epoch = 0
    if config.load_model != '':
        net, optimizer, start_epoch = load_model(
            net, config.load_model, optimizer, config.resume, config.lr)

    Trainer = train_factory[config.task]
    trainer = Trainer(config, net, optimizer)
    trainer.set_device(config, device)
    train.freeze()
    train_dataloader = prepare_dataset(config)

  # print(net.named_children())
  # Freezing the first few layers. Here I am freezing the first 7 layers ct = 0
#   ct = 0
#    for name, child in net.named_children():
#         for name2, params in net.named_parameters():
#             if config.gt:
#                 if ct > config.num_freeze_layers*2:
#                     print("Freezing layer:", name2)
#                     params.requires_grad = False
#             else:
#                 if ct < config.num_freeze_layers*2:
#                     print("Freezing layer:", name2)
#                     params.requires_grad = False
#             ct += 1

    # print(net.parameters())

    loss_history = []
    # loss = torch.Tensor([[1e10]])
    min_loss = 1e10

    # Multiple gpus support
    # chunk_sizes = config.batch_size // len(config.gpus)
    # if len(config.gpus) > 1:
    #     net = DataParallel(
    #         net, device_ids=config.gpus,
    #         chunk_sizes=chunk_sizes).to(device)
    # else:
    #     net = net.to(device)
    if config.load_model != '' and config.resume:
        model_best_loc = model_last_loc = config.load_model
    else:
        model_best_loc = os.path.join(config.save_dir, 'model_best.pth')
        model_last_loc = os.path.join(config.save_dir, 'model_last.pth')

    print('Starting training...')
    # total_iterations = len(train_dataloader)
    with trange(start_epoch + 1, config.num_epochs + 1) as iterations:
        for epoch in iterations:
            for i, data in enumerate(train_dataloader):
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(
                    device), label.to(device)
                optimizer.zero_grad()
                output1, output2 = net(img0, img1)
                loss = loss_criterion(output1, output2, label)
                loss.backward()
                optimizer.step()

                iterations.set_description(
                    'Epoch {0}[{1}/{2}]'.format(epoch, i + 1, total_iterations))
                iterations.set_postfix(Loss=loss.item())

            if min_loss > loss.item():
                min_loss = loss.item()
                save_model(model_best_loc, epoch, net, optimizer)
            save_model(model_last_loc, epoch, net, optimizer)
            loss_history.append(loss.item())
            save_plot(list(range(1, epoch + 1)),
                      loss_history, config.plot_name)


def main(config):
    logger = Logger(config)
    train_dataloader = prepare_dataset(config)
    train(train_dataloader, config, logger)
    logger.close()


if __name__ == '__main__':
    config = Config().parse()
    main(config)
