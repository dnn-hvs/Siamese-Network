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
from dataset.rdms import Rdms
from train.trainer import Trainer


def prepare_dataset(config):
    rdm = Rdms(config)
    siamese_dataset = SiameseNetworkDataset(rdm=rdm.prepare_rdms(),
                                            transform=transforms.Compose([transforms.Resize((300, 300)),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                                                               0.229, 0.224, 0.225])
                                                                          ]), should_invert=False, apply_foveate=config.foveate)

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  batch_size=config.batch_size,
                                  drop_last=True)
    return train_dataloader


def get_name(config):
    name = "_"
    name += config.task
    name += "_"
    name += config.region
    name += "_"
    name += "fov" if config.foveate else "nofov"
    name += "_"
    name += "unfrozen" if config.num_freeze_layers == 0 else "frozen"
    return name


def train(train_dataloader, config, logger):
    net = SiameseNetwork(config)

    if config.optim == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr)
    elif config.optim == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     net.parameters()), lr=config.lr, momentum=0.78)

    # Resume the model if needed
    start_epoch = 0
    if config.load_model != '':
        net, optimizer, start_epoch = load_model(
            net, config.load_model, optimizer, config.resume, config.lr)

    trainer = Trainer(config, net, optimizer, logger)
    trainer.set_device()
    trainer.freeze()

    loss_history = []
    min_loss = 1e10

    if config.load_model != '' and config.resume:
        model_best_loc = model_last_loc = config.load_model
    else:
        model_best_loc = os.path.join(
            config.save_dir, config.arch+get_name(config)+'_best.pth')
        model_last_loc = os.path.join(
            config.save_dir, config.arch+get_name(config)+'_last.pth')
    try:
        print('Starting training...')
        txt = '='*5+'Starting training...'+'='*5+'\n'
        txt += "*"*25 + "\n"
        args = dict((name, getattr(config, name)) for name in dir(config)
                    if not name.startswith('_'))
        for k, v in sorted(args.items()):
            txt += '{}: {}\n' .format(str(k), str(v))
        txt += "*"*25 + "\n"
        logger.write(txt, False)
        for epoch in tqdm(range(start_epoch + 1, config.num_epochs + 1), desc='Train'):
            loss = trainer.train(epoch, train_dataloader)
            if min_loss > loss:
                min_loss = loss
                save_model(model_best_loc, epoch, net, optimizer)
            save_model(model_last_loc, epoch, net, optimizer)
            loss_history.append(loss)
            logger.write('Epoch {0}: Loss = {1}\n'.format(epoch, loss), False)
        print('\n\nSaving plot')
        save_plot(list(range(1, len(loss_history) + 1)),
                  loss_history, config.save_dir)
        logger.write('='*5+'End training...'+'='*5, False)
    except KeyboardInterrupt:
        print('\n\nSaving plot')
        save_plot(list(range(1, len(loss_history) + 1)),
                  loss_history, config.save_dir)
        logger.write('Please check error log for more details.\n' +
                     '='*5+'End training...'+'='*5, False)
        logger.error(KeyboardInterrupt)
        print('Byeee...')
    except Exception as exception:
        logger.write('Please check error log for more details.\n' +
                     '='*5+'End training...'+'='*5, False)
        logger.error(exception)


def main(config):
    logger = Logger(config)
    train_dataloader = prepare_dataset(config)
    train(train_dataloader, config, logger)
    logger.close()


if __name__ == '__main__':
    config = Config().parse()
    main(config)
