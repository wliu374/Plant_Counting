import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random
import numpy as np
from PIL import Image
import cv2
import h5py
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from skimage import util
from skimage.measure import label
from skimage.measure import regionprops
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
# from V3liteNet import *

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr

class MaizeDataset(Dataset):
    def __init__(self, data_dir,data_list,imgfile,labelfile,  train=True, transform=None):
        self.data_dir = data_dir
        self.imgfile = imgfile
        self.labelfile = labelfile
        self.data_list = [name.split('\t') for name in open(data_dir+"/"+data_list).read().splitlines()]
        # self.ratio = ratio
        self.train = train
        self.transform = transform
        self.image_list = []

        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}

    def readfile(self,data):
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
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] not in self.images:
            ## import images and pre-labeled annotations
            image = read_image(self.data_dir + self.imgfile + file_name[0] + ".JPG")
            annotation  = open(self.data_dir + self.labelfile + file_name[0] + ".txt")
            points, gtcount = self.readfile(annotation)

            # generate ground truth density map
            h, w = image.shape[:2]
            target = np.zeros((h,w),dtype=np.float32)
            dotimage = image.copy()
            for pt in points:
                a,b = int(pt[1]),int(pt[0])
                target[a,b] = 1
                cv2.circle(dotimage, (b,a), 3, (255,0, 0), -1)

            # plt.imshow(dotimage)
            # plt.axis("off")
            # plt.savefig('results/dotimages/' + file_name[0], dpi=300)
            # plt.show()

            ## The parameter sigma is quite important
            target = gaussian_filter(target, sigma = 30)

            self.images.update({file_name[0]: image.astype('float32')})
            self.targets.update({file_name[0]: target})
            self.gtcounts.update({file_name[0]: gtcount})
            self.dotimages.update({file_name[0]: dotimage})

        sample = {
            'image': self.images[file_name[0]], ## original plot
            'target': self.targets[file_name[0]], ## target density map
            'gtcount': self.gtcounts[file_name[0]] ## ground truth counting number
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop(object):
    def __init__(self, output_size,crop_num):
        assert isinstance(output_size, (int, tuple)) # assert output_size is a integer or tuple
        self.output_size = output_size
        self.crop_num = crop_num

    def __call__(self, sample):
        image_list = []
        target_list = []
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        h, w = image.shape[:2]
        for i in range(self.crop_num):
            if isinstance(self.output_size, tuple): # if outputsize is a tuple:
                new_h = min(self.output_size[0], h)
                new_w = min(self.output_size[1], w) # to avoid output_size less than or equal to h,w
                assert (new_h, new_w) == self.output_size
            else:
                crop_size = min(self.output_size, h, w) # if outputsize is a integer
                assert crop_size == self.output_size
                new_h = new_w = crop_size #square sample

            if gtcount > 0: #ground truth count
                ## limit the area of random crop
                mask = target > 0
                ch, cw = int(np.ceil(new_h / 2)), int(np.ceil(new_w / 2))
                mask_center = np.zeros((h, w), dtype=np.uint8)
                mask_center[ch:h - ch + 1, cw:w - cw + 1] = 1
                mask = (mask & mask_center)
                idh, idw = np.where(mask == 1)
                if len(idh) != 0:
                    ids = random.choice(range(len(idh)))
                    hc, wc = idh[ids], idw[ids]
                    top, left = hc - ch, wc - cw
                else:
                    top = np.random.randint(0, h - new_h + 1)
                    left = np.random.randint(0, w - new_w + 1)
            else:
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)

            image_list.append(image[top:top + new_h, left:left + new_w, :])
            target_list.append(target[top:top + new_h, left:left + new_w])
        return {'image': image_list, 'target': target_list, 'gtcount': gtcount}

class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image_list, target_list, gtcount = sample['image'], sample['target'], sample['gtcount']
        length = len(image_list)
        for i in range(length):
            do_mirror = np.random.randint(2)
            if do_mirror:
                image_list[i] = cv2.flip(image_list[i], 1)
                target_list[i] = cv2.flip(target_list[i], 1)
        return {'image': image_list, 'target': target_list, 'gtcount': gtcount}

class Normalize(object):

    def __init__(self, scale, mean, std,train = True):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.train = train
    def __call__(self, sample):
        if self.train ==False:
            image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
            image, target = image.astype('float32'), target.astype('float32')

            # pixel normalization
            image = (self.scale * image - self.mean) / self.std
            image, target = image.astype('float32'), target.astype('float32')
            return {'image': image, 'target': target, 'gtcount': gtcount}
        else:
            image_list,target_list,gtcount = sample['image'], sample['target'], sample['gtcount']
            length = len(image_list)
            for i in range(length):
                image_list[i], target_list[i] = image_list[i].astype("float32"), target_list[i].astype("float32")
                image_list[i] = (self.scale * image_list[i] - self.mean) / self.std
                image_list[i], target_list[i] = image_list[i].astype("float32"), target_list[i].astype("float32")
            return {'image': image_list, 'target': target_list, 'gtcount': gtcount}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,train):
        self.train = train

    def __call__(self, sample):
        if self.train == False:
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
            image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
            image = image.transpose((2, 0, 1))
            target = np.expand_dims(target, axis=2)
            target = target.transpose((2, 0, 1))
            image, target = torch.from_numpy(image), torch.from_numpy(target)
            return {'image': image, 'target': target, 'gtcount': gtcount}
        else:
            image_list, target_list, gtcount = sample['image'], sample['target'], sample['gtcount']
            length = len(image_list)
            for i in range(length):
                image_list[i] = image_list[i].transpose((2, 0, 1))
                target_list[i] = np.expand_dims(target_list[i], axis=2)
                target_list[i] = target_list[i].transpose((2, 0, 1))
                image_list[i], target_list[i] = torch.from_numpy(image_list[i]), torch.from_numpy(target_list[i])
            return {'image': image_list, 'target': target_list, 'gtcount': gtcount}

class ZeroPadding(object):
    def __init__(self, psize=32,train = True):
        self.psize = psize
        self.train = train
    def __call__(self, sample):
        psize = self.psize
        if self.train == False:
            image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
            h, w = image.size()[-2:]
            ph, pw = (psize - h % psize), (psize - w % psize)
            # print(ph,pw)
            (pl, pr) = (pw // 2, pw - pw // 2) if pw != psize else (0, 0)
            (pt, pb) = (ph // 2, ph - ph // 2) if ph != psize else (0, 0)
            if (ph != psize) or (pw != psize):
                tmp_pad = [pl, pr, pt, pb]
                # print(tmp_pad)
                image = F.pad(image, tmp_pad)
                target = F.pad(target, tmp_pad)
            return {'image': image, 'target': target, 'gtcount': gtcount}
        else:
            image_list,target_list,gtcount = sample['image'], sample['target'], sample['gtcount']
            length = len(image_list)
            for i in range(length):
                h, w = image_list[i].size()[-2:]
                ph, pw = (psize - h % psize), (psize - w % psize)
                (pl, pr) = (pw // 2, pw - pw // 2) if pw != psize else (0, 0)
                (pt, pb) = (ph // 2, ph - ph // 2) if ph != psize else (0, 0)

                if (ph != psize) or (pw != psize):
                    tmp_pad = [pl, pr, pt, pb]
                    # print(tmp_pad)
                    image_list[i] = F.pad(image_list[i], tmp_pad)
                    target_list[i] = F.pad(target_list[i], tmp_pad)
            return {'image': image_list, 'target': target_list, 'gtcount': gtcount}

if __name__ == "__main__":
    data_dir = "dataset"
    train_list = "train.txt"
    val_list = "val.txt"
    # normalization
    image_scale = 1. / 255
    image_mean = [0.0210, 0.0193, 0.0181]
    image_std = [1, 1, 1]
    image_mean = np.array(image_mean).reshape((1, 1, 3))
    image_std = np.array(image_std).reshape((1, 1, 3))
    crop_num = 8
    input_size = 64
    output_stride = 8

    # model-related parameters
    optimizer = 'sgd'
    batch_size = 8
    crop_size = (256, 256)
    learning_rate = 0.01
    # milestones=[200,500]
    momentum = 0.95
    mult = 1
    num_epoch = 1000
    weight_decay = 0.0005
    mae_max = 10000
    train_transforms = transforms.Compose([
        RandomCrop(crop_size,4),
        RandomFlip(),
        Normalize(scale=image_scale, std=image_std, mean=image_mean,train = True),
        ToTensor(train=True),
        ZeroPadding(output_stride,train=True)

    ])
    val_transforms = transforms.Compose([
        Normalize(scale=image_scale, std=image_std, mean=image_mean,train=False),
        ToTensor(train = False),
        ZeroPadding(output_stride,train = False)
    ])
    trainset = MaizeDataset(
        data_dir=data_dir,
        data_list=train_list,
        imgfile="/image/",
        labelfile="/label/",
        train=True,
        transform=train_transforms
    )
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    valset = MaizeDataset(
        data_dir=data_dir,
        data_list=val_list,
        imgfile="/image/",
        labelfile="/label/",
        train=False,
        transform=val_transforms
    )
    val_loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    for i, sample in enumerate(val_loader):
        print(i)
    for i, sample in enumerate(train_loader):
        print(i)

