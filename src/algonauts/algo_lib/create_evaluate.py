# from algo_lib.feature_extract.create_RDMs import CreateRDMs


# This script generates RDMs from the activations of a DNN
# Input
#   --feat_dir : directory that contains the activations generated using generate_features.py
#   --save_dir : directory to save the computed RDM
#   --dist : dist used for computing RDM (e.g. 1-Pearson's R)
#   Note: If you have generated activations of your models using your own code, please replace
#      -   "get_layers_ncondns" to get num of layers, layers list and num of images, and
#      -   "get_features" functions to get activations of a particular layer (layer) for a particular image (i).
# Output
#   Model RDM for the representative layers of the DNN.
#   The output RDM is saved in two files submit_MEG.mat and submit_fMRI.mat in a subdirectory named as layer name.

import json
import glob
import os
import numpy as np
import datetime
import scipy.io as sio
import argparse
import zipfile
from tqdm import tqdm
import scipy
import torch.nn.functional as F
import torch
from algo_lib.evaluation.evaluate_results import Evaluate
import pandas as pd
# from utils import zip


class CreateAndEvaluateRDMs():
    def __init__(self, config):
        self.config = config

    def get_layers_ncondns(self, feat_dir):
        """
        to get number of representative layers in the DNN,
        and number of images(conditions).
        Input:
        feat_dir: Directory containing activations generated using generate_features.py
        Output:
        num_layers: number of layers for which activations were generated
        num_condns: number of stimulus images
        PS: This function is specific for activations generated using generate_features.py
        Write your own function in case you use different script to generate features.
        """
        activations = glob.glob(feat_dir + "/*" + ".mat")
        num_condns = len(activations)
        feat = sio.loadmat(activations[0])
        num_layers = 0
        layer_list = []
        for key in feat:
            if "__" in key:
                continue
            else:
                num_layers += 1
                layer_list.append(key)
        return num_layers, layer_list, num_condns

    def get_features(self, feat_dir, layer_id, i):
        """
        to get activations of a particular DNN layer for a particular image

        Input:
        feat_dir: Directory containing activations generated using generate_features.py
        layer_id: layer name
        i: image index

        Output:
        flattened activations

        PS: This function is specific for activations generated using generate_features.py
        Write your own function in case you use different script to generate features.
        """
        activations = glob.glob(feat_dir + "/*" + ".mat")
        activations.sort()
        feat = sio.loadmat(activations[i])[layer_id]
        return feat.ravel()

    def create_rdm(self, save_dir, feat_dir):
        """
        Main function to create RDM from activations
        Input:
        feat_dir: Directory containing activations generated using generate_features.py
        save_dir : directory to save the computed RDM
        dist : dist used for computing RDM (e.g. 1-Pearson's R)

        Output (in submission format):
        The model RDMs for each layer are saved in
            save_dir/layer_name/submit_fMRI.mat to compare with fMRI RDMs
            save_dir/layer_name/submit_MEG.mat to compare with MEG RDMs
        """

        # get number of layers and number of conditions(images) for RDM
        num_layers, layer_list, num_condns = self.get_layers_ncondns(feat_dir)
        # print(num_layers, layer_list, num_condns)
        cwd = os.getcwd()

        # loops over layers and create RDM for each layer
        res = {}
        for layer in tqdm(range(num_layers)):
            os.chdir(cwd)
            # RDM is num_condnsxnum_condns matrix, initialized with zeros
            RDM = np.zeros((num_condns, num_condns))

            # save path for RDMs in  challenge submission format
            layer_id = layer_list[layer]
            # RDM loop
            for i in range(num_condns):
                for j in range(num_condns):
                    # get feature for image index i and j
                    feature_i = self.get_features(feat_dir, layer_id, i)
                    feature_j = self.get_features(feat_dir, layer_id, j)

                    # compute distance 1-Pearson's R
                    if self.config["distance"] == 'pearson':
                        RDM[i, j] = 1 - \
                            np.corrcoef(feature_i, feature_j)[0][1]
                    elif self.config["distance"] == 'kernel':
                        f_i = feature_i / (feature_i*feature_i).sum()**0.5
                        f_j = feature_j / (feature_j*feature_j).sum()**0.5

                        dist = torch.dist(
                            torch.tensor(f_i.reshape(-1, 1)), torch.tensor(f_j.reshape(-1, 1)))
                        # print("Distance: ", dist)
                        RDM[i, j] = 1-scipy.exp(- dist**2 / 2).item()
                        # print("RDM[i, j]: ", RDM[i, j])

                    else:
                        print(
                            "The", self.config["distance"], "distance measure not implemented, please request through issues")

            # saving RDMs in challenge submission format
            rdm_fmri = {}
            rdm_meg = {}
            rdm_fmri['EVC_RDMs'] = RDM
            rdm_fmri['IT_RDMs'] = RDM
            rdm_meg['MEG_RDMs_late'] = RDM
            rdm_meg['MEG_RDMs_early'] = RDM
            # evaluate_rdm

            res[layer_id] = Evaluate(self.config).run(
                {'fmri': rdm_fmri, 'meg': rdm_meg})
        return res

    def run(self):
        df_78 = pd.DataFrame(columns=["epoch", "layer", "EVC %",
                                      "IT%",  "Early %", "Late %"])
        df_92 = pd.DataFrame(columns=["epoch", "layer", "EVC %",
                                      "IT%",  "Early %", "Late %"])
        df_118 = pd.DataFrame(columns=["epoch", "layer", "EVC %",
                                       "IT%",  "Early %", "Late %"])
        for image_set in self.config["image_sets"]:
            self.config["image_set"] = image_set
            feats_dir = os.path.join(
                self.config["feat_dir"], image_set+"images_feats")
            for subdir, dirs, files in os.walk(feats_dir):
                if len(dirs) == 0 and len(files) != 0:
                    save_dir = os.path.join(
                        self.config["rdms_dir"], self.config["distance"], image_set+"images_rdms", subdir.split("/")[-1])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    res = self.create_rdm(save_dir, subdir)
                    for key, val in res.items():
                        if image_set == "78":
                            df_78.loc[df_78.shape[0]] = [self.config["epoch"], key,
                                                         val["EVC %"], val["IT %"],
                                                         val["Early %"], val["Late %"]]
                        if image_set == "92":
                            df_92.loc[df_92.shape[0]] = [self.config["epoch"], key,
                                                         val["EVC %"], val["IT %"],
                                                         val["Early %"], val["Late %"]]
                        if image_set == "118":
                            df_118.loc[df_118.shape[0]] = [self.config["epoch"], key,
                                                           val["EVC %"], val["IT %"],
                                                           val["Early %"], val["Late %"]]

        return df_78, df_92, df_118