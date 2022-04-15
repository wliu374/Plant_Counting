import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks

import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks
import torch.optim as optim

# from V3litedataset import *
# from V3liteNet import *
# from V3segplus import *

from V3segdataset import *
from V3seg_net import *
from IntegrateNet import *
from Netdataset import *
from error import  *

def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr

def readfile(data):
    # data = open(filename)
    lines = data.readlines()
    X = []
    X_row = 0
    for line in lines:
        # list = line.strip('\n').split(' ')
        linedata = line.strip('\n')
        if len(linedata) == 0:
            break
        X.append(line.strip('\n').split(','))
        X_row += 1
    return X, X_row

if __name__ == "__main__":

    data_dir = "dataset"
    val_list = "val.txt"
    image_scale = 1. / 255
    # image_mean = [0.01, 0.01, 0.01]
    image_mean = [0.0210, 0.0193, 0.0181]
    image_std = [1, 1, 1]
    image_mean = np.array(image_mean).reshape((1, 1, 3))
    image_std = np.array(image_std).reshape((1, 1, 3))
    output_stride = 8

    # Load val datset
    val_transforms = transforms.Compose([
        Normalize(scale=image_scale, std=image_std, mean=image_mean, train=False),
        ToTensor(train=False),
        ZeroPadding(output_stride, train=False)
    ])
    valset = MaizeTasselDataset(
        data_dir=data_dir, data_list=val_list, imgfile="/image/", labelfile="/label/",train=False,transform=val_transforms
    )
    val_loader = DataLoader(
        valset, batch_size=1,shuffle=False,num_workers=0,pin_memory=True)

    """
    Here we take V3seg and IntegrateNet as examples for visualization
    """

    ### v3seg
    path = "results/models/v3seg/model_best.pth.tar"
    # file_name = "model_best.pth/archive/data.pkl"
    load_net = V3seg()
    load_net = nn.DataParallel(load_net)
    checkpoint = torch.load(path)
    load_net.load_state_dict(checkpoint['state_dict'])
    load_net.eval()
    image_list = valset.image_list

    pdcounts = []
    gtcounts = []

    for i, sample in enumerate(val_loader):
        ## predict
        input_image = sample['image']
        gtcount = sample['gtcount']
        dic = load_net(input_image.cuda(),is_normalize = False)
        R = dic['R']
        R = Normalizer.gpu_normalizer(R)
        R = np.clip(R,0,None)
        pdcount = R.sum()
        gtcount = float(gtcount.numpy())
        pdcounts.append(pdcount)
        gtcounts.append(gtcount)

    print(compute_mae(pdcounts,gtcounts),compute_rmse(pdcounts,gtcounts),rsquared(pdcounts, gtcounts))
    xlim = np.arange(50,110,0.1)
    ylim = np.arange(50,110,0.1)
    font  = {'family':'Times New Roman','size' :17}
    plt.scatter(gtcounts,pdcounts,alpha=0.6)
    plt.plot(xlim,ylim,linewidth = 2,linestyle = ':')
    # plt.rcParams['figure.figsize'] = (6,3)
    # plt.xlabel('gt_counts',font = font)
    # plt.ylabel("pd_counts",font = font)
    plt.xticks(font = font)
    plt.yticks(font = font)
    plt.show()

    print(pdcounts)
    print(gtcounts)

    ### IntegrateNet
    for i, sample in enumerate(val_loader):
        ## predict
        input_image = sample['image']
        gtcount = sample['gtcount']
        dic = load_net(input_image)
        output = dic['density']
        output = output.squeeze().detach().numpy()
        output = np.clip(output,0,None)
        pdcount = output.sum()
        gtcount = float(gtcount.numpy())
        pdcounts.append(pdcount)
        gtcounts.append(gtcount)

    print(compute_mae(pdcounts,gtcounts),compute_rmse(pdcounts,gtcounts),rsquared(pdcounts, gtcounts))
    xlim = np.arange(50,110,0.1)
    ylim = np.arange(50,110,0.1)
    font  = {'family':'Times New Roman','size' :17}
    plt.scatter(gtcounts,pdcounts,alpha=0.6)
    plt.plot(xlim,ylim,linewidth = 2,linestyle = ':')
    # plt.rcParams['figure.figsize'] = (6,3)
    # plt.xlabel('gt_counts',font = font)
    # plt.ylabel("pd_counts",font = font)
    plt.xticks(font = font)
    plt.yticks(font = font)
    plt.show()

    print(pdcounts)
    print(gtcounts)





