from algo_lib.feature_extract.generate_features import GenerateFeatures
from algo_lib.create_evaluate import CreateAndEvaluateRDMs

from algo_lib.utils.config import Config
from argparse import Namespace
import argparse
import os
import torch


class Algonauts:
    def __init__(self, exp_id, arch, load_model, epoch):
        args = {}
        args['distance'] = "pearson"
        args["exp_id"] = exp_id
        args["arch"] = arch
        args["load_model"] = load_model
        args["image_dir"] = "../data"
        args["feat_dir"] = os.path.join("./"+exp_id, "feats")
        args["rdms_dir"] = os.path.join("./"+exp_id, "rdms")
        args["res_dir"] = os.path.join("./"+exp_id, "results")

        args["image_sets"] = ["92", "118", "78"]
        args["epoch"] = epoch

        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        args["device"] = torch.device('cpu')

        if not os.path.exists(args["res_dir"]):
            os.makedirs(args["res_dir"])
        if not os.path.exists(args["rdms_dir"]):
            os.makedirs(args["rdms_dir"])
        if not os.path.exists(args["feat_dir"]):
            os.makedirs(args["feat_dir"])

        self.config = args

    def run(self):
        GenerateFeatures(self.config).run()
        return CreateAndEvaluateRDMs(self.config).run()
