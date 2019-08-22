import argparse
from datetime import datetime
import os


class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--train_dir', default='../data_the_data/',
                                 help='Training Dataset Directory')
        self.parser.add_argument('--test_dir', default='../data/Test_Data/',
                                 help='Test Dataset Directory')

        self.parser.add_argument(
            '-f', '--foveate', action='store_true', help="Use foveated Wavelet Transform on images")
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                 'Reloaded the optimizer parameter and '
                                 'set load_model to model_last.pth ')
        self.parser.add_argument(
            '--evc', action='store_true', help="Use 'EVC' values of the RDMs. If not mentioned, it uses 'IT' values of RDMs")

        # model
        self.parser.add_argument('-a', '--arch', default='alexnet',
                                 help='model architecture. alexnet | vgg-16 | inception')
        self.parser.add_argument('--optim', default='adam',
                                 help='Optimiser used for training. adam|sgd')

        # train
        self.parser.add_argument('--lr', type=float, default=0.005,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_epochs', type=int, default=78,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='batch size')

        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help='Number of workers')

        self.parser.add_argument('--plot_name', default='Graph',
                                 help='Name of the loss plot that will be saved. Default name is Graph.png')

        self.parser.add_argument('--num_freeze_layers', type=int, default=0,
                                 help='Number of layers to freeze')

        self.parser.add_argument('-gt', action='store_true',
                                 help='If true, the layers greater than num_freeze_layers will be frozen , else \
                                        the  layers lesser than num_freeze_layers will be frozen')

        # self.parser.add_argument('--lr_step', type=str, default='90,120',
        #                          help='drop learning rate by 10.')

        # self.parser.add_argument('--master_batch_size', type=int, default=-1,
        #                          help='batch size on the master gpu.')
        # self.parser.add_argument('--num_iters', type=int, default=-1,
        #                          help='default: #samples / batch_size.')
        # self.parser.add_argument('--val_intervals', type=int, default=5,
        #                          help='number of epochs to run validation.')
        # self.parser.add_argument('--trainval', action='store_true',
        #                          help='include validation in training and '
        #                               'test on test set')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(
            len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        opt.save_dir = os.path.join('../models', opt.arch + "_" +
                                    str(datetime.now().strftime(
                                        "%d-%b-%y--%X")))
        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt
