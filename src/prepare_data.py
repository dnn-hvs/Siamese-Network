from itertools import combinations
import os
from shutil import copy

# os.mkdir('../data_the_data')
image_92_path = '../data/Training_Data/92_Image_Set/92images/'
image_118_path = '../data/Training_Data/118_Image_Set/118images/'
images_92 = os.listdir(image_92_path)
images_118 = os.listdir(image_118_path)
print(len(images_118), len(images_92))

combs_92 = list(combinations(images_92, 2))
combs_118 = list(combinations(images_118, 2))
tot_list = combs_92 + combs_118

for i, val in enumerate(combs_92):
    new_dir = '../data_the_data/92_images/' + str(i)
    os.mkdir(new_dir)
    copy(image_92_path+val[0], new_dir)
    copy(image_92_path+val[1], new_dir)


for i, val in enumerate(combs_118):
    new_dir = '../data_the_data/118_images/' + str(i)
    os.mkdir(new_dir)
    copy(image_118_path+val[0], new_dir)
    copy(image_118_path+val[1], new_dir)
