import sys
import numpy as np
from itertools import combinations
import os
from shutil import copy
import cv2
from tqdm import tqdm


def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d


def pyramid(im, sigma=1, prNum=6):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]

    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)

    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)

    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids


def foveat_img(im, fixs):
    """
    im: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    This function outputs the foveated image with given input image and fixations.
    """
    sigma = 0.248
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape

    # compute coef
    p = 10
    k = 3
    alpha = 2.5

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt(
            (x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)

    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma

    omega[omega > 1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))

    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    # print('num of full-res pixel', np.sum(Ms[0] == 1))
    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])

    im_fov = im_fov.astype(np.uint8)
    return im_fov


try:
    os.mkdir('../data_foveated')
    os.mkdir('../data_foveated/92_images')
    os.mkdir('../data_foveated/118_images')

except FileExistsError:
    pass
image_92_path = '../data/Training_Data/92_Image_Set/92images/'
image_118_path = '../data/Training_Data/118_Image_Set/118images/'
images_92 = os.listdir(image_92_path)
images_118 = os.listdir(image_118_path)
print(len(images_118), len(images_92))

combs_92 = list(combinations(images_92, 2))
combs_118 = list(combinations(images_118, 2))
tot_list = combs_92 + combs_118

for i, val in enumerate(tqdm(combs_92)):
    new_dir = '../data_foveated/92_images/' + str(i)
    os.mkdir(new_dir)
    image = cv2.imread(image_92_path+val[0])
    xc, yc = int(image.shape[1]/2), int(image.shape[0]/2)
    image = foveat_img(image, [(xc, yc)])
    cv2.imwrite(os.path.join(new_dir, val[0]), image)

    image = cv2.imread(image_92_path+val[1])
    xc, yc = int(image.shape[1]/2), int(image.shape[0]/2)
    image = foveat_img(image, [(xc, yc)])
    cv2.imwrite(os.path.join(new_dir, val[1]), image)


for i, val in enumerate(tqdm(combs_118)):
    new_dir = '../data_foveated/118_images/' + str(i)
    os.mkdir(new_dir)
    image = cv2.imread(image_118_path+val[0])
    xc, yc = int(image.shape[1]/2), int(image.shape[0]/2)
    image = foveat_img(image, [(xc, yc)])
    cv2.imwrite(os.path.join(new_dir, val[0]), image)

    image = cv2.imread(image_118_path+val[1])
    xc, yc = int(image.shape[1]/2), int(image.shape[0]/2)
    image = foveat_img(image, [(xc, yc)])
    cv2.imwrite(os.path.join(new_dir, val[1]), image)
