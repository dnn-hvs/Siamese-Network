import argparse
import os
# functions to generate features in utils.py
from lib.utils import utils, config, networks_factory, constants
import torch
import glob
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
import scipy.io as sio


class GenerateFeatures():
    def __init__(self, config):
        self.config = config

    def execute_model(self, model, feats_save_dir):
        if torch.cuda.is_available():
            model.to(self.config.device)
        model.eval()
        centre_crop = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_list = glob.glob(self.config.image_dir + "/*.jpg")
        image_list.sort()
        image_list = image_list
        for image in tqdm(image_list):
            img = Image.open(image)
            filename = image.split("/")[-1].split(".")[0]
            input_img = Variable(centre_crop(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img = input_img.to(self.config.device)

            x = model.forward(input_img)
            save_path = os.path.join(feats_save_dir, filename+".mat")
            feats = OrderedDict()
            for key, value in x.items():
                feats[key] = value.data.cpu().numpy()
            sio.savemat(save_path, feats)

    def get_model(self, model, load_model):
        if self.config.load_model is not None or self.config.fullblown or self.config.generate_features:
            model = utils.get_model(model)
            return utils.load_model(model, load_model)
        else:
            return utils.get_model(self.config.arch)

    def get_model_full_name(self, name):
        split_name = name.split(constants.UNDER_SCORE)
        model_name = self.get_model_name(name)
        prev_combinations = split_name[1] if "sqnet" not in name else split_name[2]
        extension = split_name[2] if "sqnet" not in name else split_name[3]
        new_combinations = []

        if prev_combinations[0] == "0" or prev_combinations[0] == "1":
            new_combinations.append(
                "fmri" if prev_combinations[0] == "0" else "meg")
            new_combinations.append(
                "early" if prev_combinations[1] == "0" else "late")
            new_combinations.append(
                "fov" if prev_combinations[2] == "0" else "nofov")
            new_combinations.append(
                "unfrozen" if prev_combinations[3] == "0" else "frozen")
            return model_name + constants.UNDER_SCORE + constants.UNDER_SCORE.join(new_combinations) + constants.UNDER_SCORE+extension
        return name

    def get_model_name(self, name):
        split_name = name.split(constants.UNDER_SCORE)
        if "sqnet" in name:
            return split_name[0]+"_"+split_name[1]
        return split_name[0]

    def run(self):
        if self.config.fullblown or self.config.generate_features:
            for image_set in self.config.image_sets:
                print("Image Set: ", image_set)
                if not self.config.test_set:
                    self.config.image_dir = os.path.join("../data/Training_Data/" +
                                                         image_set+"_Image_Set", image_set+"images")
                else:
                    self.config.image_dir = os.path.join(
                        "../data/Test_Data", image_set+"images")
                models_list = glob.glob(self.config.models_dir + "/*.pth")
                for model_pth in models_list:
                    pth_name = model_pth.split(constants.FORWARD_SLASH)[-1]
                    pth_name = self.get_model_full_name(pth_name)
                    model_name = self.get_model_name(pth_name)
                    subdir_name = pth_name.split(".")[0]
                    print("model_name: ", model_name,
                          " subdir_name: ", subdir_name)
                    path = os.path.join(
                        self.config.feat_dir, image_set+"images_feats", subdir_name)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    model = self.get_model(model_name, model_pth)
                    self.execute_model(model, path)
            return
