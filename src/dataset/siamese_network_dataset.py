from torch.utils.data import DataLoader, Dataset
import random
import cv2
import torch
import numpy as np
from PIL import Image
from utils import foveate


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, rdm, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
        self.rdm = rdm

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        while True:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] == img1_tuple[1]:
                break

        img0 = np.asarray(Image.open(img0_tuple[0]).convert("L"))
        img1 = np.asarray(Image.open(img1_tuple[0]).convert("L"))
        img0 = foveate.foveat_img(
            img0, [[int(img0.shape[1]/2), int(img0.shape[0]/2)]])
        img1 = foveate.foveat_img(
            img1, [[int(img1.shape[1]/2), int(img1.shape[0]/2)]])
        img0 = Image.fromarray(np.uint8(img0))
        img1 = Image.fromarray(np.uint8(img1))

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, self.__get_rdm_pair__(img0_tuple[0], img1_tuple[0])

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

    def __get_rdm_pair__(self, img1_name, img2_name):
        img_num1 = int(img1_name.split("/image_")[1].split(".jpg")[0])
        img_num2 = int(img2_name.split("/image_")[1].split(".jpg")[0])

        if(img1_name.find("92_Image_Set") != -1):
            rdm = self.rdm[92]
        else:
            rdm = self.rdm[118]
        return (torch.from_numpy(np.array([rdm[img_num1-1][img_num2-1]], dtype=np.float32)))
