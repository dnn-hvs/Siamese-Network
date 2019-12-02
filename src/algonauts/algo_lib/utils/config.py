import argparse
from datetime import datetime
import os
import algo_lib.utils.networks_factory as networks_factory
import algo_lib.utils.constants as constants
import torch


class Config(object):
    def __init__(self):
        RDM_distance_choice = ['pearson', 'kernel']

        self.parser = argparse.ArgumentParser(
            description='generate DNN activations from a stimuli dir')
        self.parser.add_argument('-d', '--distance', help='distance for RDMs',
                                 default="pearson", choices=RDM_distance_choice)
        self.parser.add_argument("--arch", help='DNN choice',
                                 default="all", choices=networks_factory.models.keys())
        self.parser.add_argument("--load_model",
                                 help='Path to the desired model to be tested of the architecture specified',
                                 default=None)
        self.parser.add_argument('--exp_id', help='Stores feats, rdms and results in this directory',
                                 default=None, type=str)

        # Directories
        self.parser.add_argument('-id', '--image_dir', help='stimulus directory path',
                                 default=None, type=str)

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(namespace=args)

        opt.image_sets = ['92', '118', '78']
        if opt.exp_id is not None:
            opt.feat_dir = os.path.join(opt.exp_id, "feats")
            opt.rdms_dir = os.path.join(opt.exp_id, "rdms")
            opt.res_dir = os.path.join(opt.exp_id, "results")
        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt
