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
from Netdataset import *
from IntegrateNet import *
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

    # # Load val datset
    val_transforms = transforms.Compose([
        Normalize(scale=image_scale, std=image_std, mean=image_mean, train=False),
        ToTensor(train=False),
        ZeroPadding(output_stride, train=False)
    ])
    valset = MaizeDataset(
        data_dir=data_dir, data_list=val_list, imgfile="/image/", labelfile="/label/",train=False,transform=val_transforms
    )
    val_loader = DataLoader(
        valset, batch_size=1,shuffle=False,num_workers=0,pin_memory=True
    )
    # #
    # # Load network
    path = "results/models/IntegrateNet/model_best.pth"
    # file_name = "model_best.pth/archive/data.pkl"
    load_net = IntegrateNet()
    checkpoint = torch.load(path)
    load_net.load_state_dict(checkpoint['state_dict'])
    load_net.eval()
    image_list = valset.image_list

    for i, sample in enumerate(val_loader):
        ## predict
        input_image = sample['image']
        dic = load_net(input_image)
        output = dic['density']
        output = output.squeeze().detach().numpy()
        output = np.clip(output, 0, None)
        pdcount = output.sum()
        print('pdcount',pdcount)

        ## ground truth
        gt_image = read_image("dataset/image/" + image_list[i] + '.jpg')
        annotations = open('dataset/label/'+image_list[i] + '.txt')
        np.save(file = "gt_image"+image_list[i]+".npy",arr = gt_image)
        points,gtcount = readfile(annotations)
        print(image_list[i]+"gtcount:",gtcount)
        h,w = gt_image.shape[:2]
        target = np.zeros((h,w),dtype=np.float32)
        dotimage = gt_image.copy()

        ## circle annotations representing the locations of plants
        print("dotimage shape:",dotimage.shape)
        for pt in points:
            a,b = int(pt[1]), int(pt[0])
            target[a,b] = 1
            cv2.circle(dotimage,(b,a),25,(255,0,0),2)

        # generate density map
        density = gaussian_filter(target,30)

        dens = output
        dens = cv2.resize(dens,(w,h),interpolation=cv2.INTER_CUBIC)

        plt.imshow(density, vmin=0, vmax=0.001)
        plt.colorbar(orientation = 'horizontal')
        plt.axis("off")
        plt.savefig('results/visualization/density/ground_truth/gt_'+image_list[i]+'.png', dpi=300)
        plt.show()
        plt.imshow(dens,vmin=0, vmax=0.001)
        plt.colorbar(orientation = 'horizontal')
        plt.axis("off")
        plt.savefig('results/visualization/density/predict/pd_'+image_list[i]+'.png', dpi=300)
        plt.show()

        cmap = plt.cm.get_cmap('jet')
        ddens = dens.copy()
        # ddens[ddens < 0.0002] = 0
        cmap_dens = ddens / (ddens.max() + 1e-30)
        cmap_dens = cmap(cmap_dens) * 255.
        printimage = 0.7 * gt_image + 0.3 * cmap_dens[:, :, 0:3]
        plt.imshow(printimage.astype(np.uint8))
        plt.axis('off')
        plt.savefig("results/visualization/density/cmap/cmap_"+image_list[i]+'.png', dpi=300)
        plt.show()
        plt.imshow(dotimage)
        plt.axis("off")
        plt.savefig('results/visualization/dotimages'+image_list[i]+'.png', dpi=300)
        plt.show()


