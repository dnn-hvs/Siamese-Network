from torch.utils.data import DataLoader, Dataset
import random
import cv2
import torch
import numpy as np
from PIL import Image
from utils import foveate
import os


class SiameseNetworkDataset(Dataset):

    def __init__(self, rdm, transform=None, should_invert=True, apply_foveate=False):
        self.transform = transform
        self.should_invert = should_invert
        self.rdm = rdm
        self.apply_foveate = apply_foveate
        self.image_list = np.array(
            [(os.path.join(dirpath, filenames[0]), os.path.join(dirpath, filenames[1])) for dirpath, dirnames, filenames in os.walk('../data_the_data') if filenames])
        # print(len(self.image_list))

    def __getitem__(self, index):
        img0_path, img1_path = random.choice(self.image_list)
        # print(img0_path, img1_path)
        img0 = self.modify_image(img0_path)
        img1 = self.modify_image(img1_path)
        return img0, img1, self.get_rdm_pair(img0_path, img1_path)

    def __len__(self):
        return len(self.image_list)

    def get_rdm_pair(self, img1_name, img2_name):
        img_num1 = int(img1_name.split("/image_")[1].split(".jpg")[0])
        img_num2 = int(img2_name.split("/image_")[1].split(".jpg")[0])
        if img1_name.find("92_images") != -1:
            rdm = self.rdm['92']
        else:
            rdm = self.rdm['118']

        return (torch.from_numpy(np.array([rdm[img_num1-1][img_num2-1]], dtype=np.float32)))

    def modify_image(self, img_path):
        img = np.asarray(Image.open(img_path).convert("L"))
        if self.apply_foveate:
            img = foveate.foveat_img(
                img, [[int(img.shape[1]/2), int(img.shape[0]/2)]])
        img = np.dstack((img, img, img))
        img = Image.fromarray(np.uint8(img))
        if self.transform is not None:
            img = self.transform(img)

        return img
