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

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os

def sign(array):
    h,w = array.shape
    for i in range(h):
        for j in range(w):
            if array[i][j]<=0:
                array[i][j] = 0
            else:
                array[i][j] = 1
    return array

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr


class MaizeTasselDataset(Dataset):
    def __init__(self, data_dir,data_list,imgfile,labelfile,  train=True, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_dir+"/"+data_list).read().splitlines()]

        self.train = train
        self.transform = transform
        self.image_list = []
        self.imgfile = imgfile
        self.labelfile = labelfile
        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}
        self.pseudomask = {}
        self.gtpseudomask = {}

    def readfile(self, data):
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
            image = read_image(self.data_dir + self.imgfile + file_name[0] + ".JPG")
            annotation = open(self.data_dir + self.labelfile + file_name[0] + ".txt")
            points, gtcount = self.readfile(annotation)
            h, w = image.shape[:2]
            target = np.zeros((h,w),dtype=np.float32)
            dotimage = image.copy()
            pseudomask = np.zeros((h+10, w+10), dtype=np.float32)
            filter = np.zeros((11,11))
            filter[1:10,1:10] = 1
            filter[0,3:8] = 1
            filter [3:8,0] = 1
            filter[10,3:8] = 1
            filter [3:8, 10] = 1
            # print("filter",filter)

            for pt in points:
                pt[1],pt[0] = int(pt[1]), int(pt[0])
                target[pt[1],pt[0]] = 1
                cv2.circle(dotimage, (pt[0],pt[1]), 16, (255,0, 0), -1)
                a,b = pt[1]+5, pt[0]+5
                pseudomask [a-5:a+6,b-5:b+6] += filter

            pseudomask = sign(pseudomask)
            pseudomask = pseudomask[5:h+5,5:w+5]

            # print(pseudomask.shape)
            target = gaussian_filter(target, 30)
            # print(np.sum(target),gtcount)

            # print(gtcount)
            # print(dotimage.shape)
            # fig = plt.figure(figsize=(16, 9))
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(image)
            # ax1.get_xaxis().set_visible(False)
            # ax1.get_yaxis().set_visible(False)

            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(dotimage)
            # ax2.get_xaxis().set_visible(False)
            # ax2.get_yaxis().set_visible(False)
            # # plt.imshow(target)
            # plt.show()

            # print(target.sum())

            self.images.update({file_name[0]: image.astype('float32')})
            self.targets.update({file_name[0]: target})
            self.gtcounts.update({file_name[0]: gtcount})
            self.pseudomask.update({file_name[0]: pseudomask})
            self.gtpseudomask.update({file_name[0]: pseudomask})
            self.dotimages.update({file_name[0]: dotimage})

        sample = {
            'image': self.images[file_name[0]],  ## original image
            'target': self.targets[file_name[0]],  ## target density map
            'gtcount': self.gtcounts[file_name[0]],  ## ground truth counting number
            'pseudomask':self.pseudomask[file_name[0]],
            'gtpseudomask': self.gtpseudomask[file_name[0]]
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
        pseudomask_list = []
        image, target, gtcount , pseudomask = sample['image'], sample['target'], sample['gtcount'],sample['pseudomask']
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
                mask = target > 0 # target??
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
            pseudomask_list.append(pseudomask[top:top + new_h, left:left + new_w])
        return {'image': image_list, 'target': target_list, 'gtcount': gtcount,'pseudomask':pseudomask_list}

class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image_list, target_list, gtcount ,pseudomask_list =\
            sample['image'], sample['target'], sample['gtcount'],sample['pseudomask']
        length = len(image_list)
        for i in range(length):
            do_mirror = np.random.randint(2)
            if do_mirror:
                image_list[i] = cv2.flip(image_list[i], 1)
                target_list[i] = cv2.flip(target_list[i], 1)
                pseudomask_list[i] = cv2.flip(pseudomask_list[i], 1)
        return {'image': image_list, 'target': target_list, 'gtcount': gtcount,'pseudomask':pseudomask_list}


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
            image_list,target_list,gtcount ,pseudomask_list= \
                sample['image'], sample['target'], sample['gtcount'],sample['pseudomask']
            length = len(image_list)
            for i in range(length):
                image_list[i], target_list[i] = image_list[i].astype("float32"), target_list[i].astype("float32")
                image_list[i] = (self.scale * image_list[i] - self.mean) / self.std
                image_list[i], target_list[i] = image_list[i].astype("float32"), target_list[i].astype("float32")
                pseudomask_list[i] = pseudomask_list[i].astype('float32')
            return {'image': image_list, 'target': target_list, 'gtcount': gtcount,'pseudomask':pseudomask_list}


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
            image_list,target_list,gtcount ,pseudomask_list= \
                sample['image'], sample['target'], sample['gtcount'],sample['pseudomask']
            length = len(image_list)
            for i in range(length):
                h, w = image_list[i].size()[-2:]
                ph, pw = (psize - h % psize), (psize - w % psize)
                # print(ph,pw)

                (pl, pr) = (pw // 2, pw - pw // 2) if pw != psize else (0, 0)
                (pt, pb) = (ph // 2, ph - ph // 2) if ph != psize else (0, 0)

                if (ph != psize) or (pw != psize):
                    tmp_pad = [pl, pr, pt, pb]
                    # print(tmp_pad)
                    image_list[i] = F.pad(image_list[i], tmp_pad)
                    target_list[i] = F.pad(target_list[i], tmp_pad)
                    pseudomask_list[i] = F.pad(pseudomask_list[i],tmp_pad)

            return {'image': image_list, 'target': target_list, 'gtcount': gtcount,'pseudomask':pseudomask_list}



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
            image_list,target_list,gtcount ,pseudomask_list= \
                sample['image'], sample['target'], sample['gtcount'],sample['pseudomask']
            length = len(image_list)
            for i in range(length):
                image_list[i] = image_list[i].transpose((2, 0, 1))
                target_list[i] = np.expand_dims(target_list[i], axis=2)
                target_list[i] = target_list[i].transpose((2, 0, 1))
                nh, nw = pseudomask_list[i].shape
                nh, nw = int(nh / 8), int(nw / 8)
                pseudomask_list[i] = np.uint8(pseudomask_list[i])
                pseudomask_list[i] = cv2.resize(pseudomask_list[i],(nh,nw),interpolation=cv2.INTER_CUBIC)
                pseudomask_list[i] = np.expand_dims(pseudomask_list[i], axis=2)
                pseudomask_list[i] = pseudomask_list[i].transpose((2, 0, 1))
                pseudomask_list[i] = pseudomask_list[i].astype("float32")
                image_list[i], target_list[i] = torch.from_numpy(image_list[i]), torch.from_numpy(target_list[i])
                pseudomask_list[i] = torch.from_numpy(pseudomask_list[i])
            return {'image': image_list, 'target': target_list, 'gtcount': gtcount,'pseudomask':pseudomask_list}

if __name__ == "__main__":
    data_dir = "dataset"
    train_list = "train.txt"
    val_list = "val.txt"
    image_scale = 1. / 255
    image_mean = [0.0210, 0.0193, 0.0181]
    image_std = [1, 1, 1]
    image_mean = np.array(image_mean).reshape((1, 1, 3))
    image_std = np.array(image_std).reshape((1, 1, 3))
    input_size = 64
    output_stride = 8
    resized_ratio = 0.125
    print_every = 1

    # model-related parameters
    optimizer = 'sgd'
    batch_size = 1
    crop_size = (256, 256)
    learning_rate = 0.000001
    milestones = [200, 400]
    momentum = 0.95
    mult = 1
    num_epoch = 600
    weight_decay = 0.0005
    train_transforms = transforms.Compose([
        RandomCrop(crop_size,4),
        RandomFlip(),
        Normalize(scale=image_scale, std=image_std, mean=image_mean,train = True),
        ToTensor(train=True),
        ZeroPadding(output_stride,train=True)

    ])

    trainset = MaizeTasselDataset(
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
    for i, sample in enumerate(train_loader):
        image_list,target_list = sample['image'], sample['target']
        length = len(image_list)
        for j in range(length):
            if i % 5 ==4:
                print(image_list[j].squeeze().size())
