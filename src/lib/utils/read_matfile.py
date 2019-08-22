from scipy import io
from scipy.spatial.distance import squareform
from scipy import stats
import os
import sys
import h5py
import numpy as np


def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


def loadnpy(npyfile):
    return np.load(npyfile)


def load(data_file):
    root, ext = os.path.splitext(data_file)
    return {'.npy': loadnpy,
            '.mat': loadmat
            }.get(ext, loadnpy)(data_file)


def get_rdm(image_set_92=1, evc=1):
    norm_rdms = {}
    norm_rdms[92] = get_norm_rdms(
        '../data/Training_Data/92_Image_Set/target_fmri.mat', evc)
    norm_rdms[118] = get_norm_rdms(
        '../data/Training_Data/118_Image_Set/target_fmri.mat', evc)
    return norm_rdms


def get_norm_rdms(file_path, evc):
    rdms = load(file_path)
    if evc:
        rdm_norm = normalise(rdms["EVC_RDMs"])
    else:
        rdm_norm = normalise(rdms["IT_RDMs"])
    return np.mean(rdm_norm, axis=0)
# Performs zscore for each subject


def normalise(input):
    input_zscore = []
    for i in range(15):
        input_zscore.append(stats.zscore(input[i], axis=None))
    return np.array(input_zscore)
