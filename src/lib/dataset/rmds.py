import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import _init_paths
import os
import sys
from datetime import datetime
from utils.read_matfile import load
from dataset.siamese_network_dataset import SiameseNetworkDataset
from utils.logger import Logger
import numpy as nu
from scipy import stats


class RDMS():
    def __init__(self, config):
        self.config = config

    def prepare_rdms(self):
        if self.config.task == 'fmri':
            return self.get_fmri_rdm(self.config.region)
        else:
            return self.get_meg_rdm(self.config.refion)

    def normalize(self):
        input_zscore = []
        for i in range(15):
            input_zscore.append(stats.zscore(input[i], axis=None))
        return np.array(input_zscore)

    def get_fmri_rdm(region):
        rdm = {}
        rdms_92 = load('../data/Training_Data/92_Image_Set/target_fmri.mat')
        rdms_118 = load('../data/Training_Data/118_Image_Set/target_fmri.mat')
        if region == 'early':
            rdm[92] = self.normalize(rdms_92['EVC_RDMs'])
            rdms[118] = self.normalize(rdms_118['EVC_RDMs'])
        else:
            rdm[92] = self.normalize(rdms_92['IT_RDMs'])
            rdms[118] = self.normalize(rdms_118['IT_RDMs'])
        rdm[92] = np.mean(rdm[92], axis=0)
        rdm[118] = np.mean(rdm[118], axis=0)
        return rdm

    def get_meg_rdm(region):
        rdm = {}
        rdms_92 = load('../data/Training_Data/92_Image_Set/target_meg.mat')
        rdms_118 = load('../data/Training_Data/118_Image_Set/target_meg.mat')
        if region == 'early':
            rdm[92] = self.normalize(rdms_92['MEG_RDMs_early'])
            rdms[118] = self.normalize(rdms_118['MEG_RDMs_early'])
        else:
            rdm[92] = self.normalize(rdms_92['MEG_RDMs_late'])
            rdms[118] = self.normalize(rdms_118['MEG_RDMs_late'])
        rdm[92] = np.mean(rdm[92], axis=0)
        rdm[118] = np.mean(rdm[118], axis=0)
        return rdm
