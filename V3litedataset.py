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
from V3liteNet import *

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr

class MaizeTasselDataset(Dataset):
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
        self.mask = {}

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
            image = read_image(self.data_dir + self.imgfile + file_name[0] + ".JPG")
            annotation  = open(self.data_dir + self.labelfile + file_name[0] + ".txt")
            points, gtcount = self.readfile(annotation)

            # annotation = pd.read_csv(self.data_dir + file_name[1])
            h, w = image.shape[:2]
            # nh = int(np.ceil(h * self.ratio))
            # nw = int(np.ceil(w * self.ratio))
            #
            # image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            # target = np.zeros((nh, nw), dtype=np.float32)
            target = np.zeros((h,w),dtype=np.float32)
            dotimage = image.copy()
            # gtcount,points=self.pointlocation(annotation)

            for pt in points:
                # print(pt)
                a,b = int(pt[1]),int(pt[0])
                # print(a)
                # print(b)
                # pt[0], pt[1] = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                # print(file_name)
                target[a,b] = 1

                cv2.circle(dotimage, (a,b), 16, (255,0, 0), -1)
            mask = target
            target = gaussian_filter(target, 30)
            # print(gtcount)
            # print(dotimage.shape)
            # plt.imshow(dotimage)
            # plt.imshow(target)
            # plt.show()
            # print(target.sum())

            self.images.update({file_name[0]: image.astype('float32')})
            self.targets.update({file_name[0]: target})
            self.gtcounts.update({file_name[0]: gtcount})
            self.dotimages.update({file_name[0]: dotimage})
            self.mask.update({file_name[0]:mask})
        sample = {
            'image': self.images[file_name[0]], ## original image
            'target': self.targets[file_name[0]], ## target density map
            'gtcount': self.gtcounts[file_name[0]], ## ground truth counting number
            'mask': self.mask[file_name[0]]
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

        return {'image_list': image_list, 'target_list': target_list, 'gtcount': gtcount}

class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image_list, target_list = sample['image_list'], sample['target_list']
        # image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        length = len(image_list)
        for i in range(length):
            do_mirror = np.random.randint(2)
            if do_mirror:
                image_list[i] = cv2.flip(image_list[i], 1)
                target_list[i] = cv2.flip(target_list[i], 1)
        return {'image_list': image_list, 'target_list': target_list}


class Normalize(object):

    def __init__(self, scale, mean, std,train = True):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.train = train

    def __call__(self, sample):
        if self.train == True:
            image_list,target_list = sample['image_list'],sample['target_list']
            length = len(image_list)
            for i in range(length):
                image_list[i] , target_list[i]= image_list[i].astype('float32'), target_list[i].astype('float32')
                image_list[i] = (self.scale * image_list[i] - self.mean) / self.std
                image_list[i] , target_list[i]= image_list[i].astype('float32'), target_list[i].astype('float32')
            return {'image_list': image_list, 'target_list': target_list}
        else:
            image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
            image, target = image.astype('float32'), target.astype('float32')

            # pixel normalization
            image = (self.scale * image - self.mean) / self.std

            image, target = image.astype('float32'), target.astype('float32')

            return {'image': image, 'target': target, 'gtcount': gtcount}


class ZeroPadding(object):
    def __init__(self, psize=32,train = True):
        self.psize = psize
        self.train = train
    def __call__(self, sample):
        psize = self.psize
        if self.train:
            image_list,target_list = sample['image_list'],sample['target_list']
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
            return {'image_list': image_list, 'target_list': target_list}

        else:
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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,train = True):
        self.train = train

    def __call__(self, sample):
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        if self.train == False:
            image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
            image = image.transpose((2, 0, 1))
            target = np.expand_dims(target, axis=2)
            target = target.transpose((2, 0, 1))
            image, target = torch.from_numpy(image), torch.from_numpy(target)
            return {'image': image, 'target': target, 'gtcount': gtcount}

        else:
            image_list, target_list = sample['image_list'], sample['target_list']
            length = len(image_list)
            for i in range(length):
                image_list[i] = image_list[i].transpose((2,0,1))
                target_list[i] = np.expand_dims(target_list[i],axis=2)
                target_list[i] = target_list[i].transpose((2,0,1))
                image_list[i], target_list[i] = torch.from_numpy(image_list[i]), torch.from_numpy(target_list[i])
            return {'image_list': image_list, 'target_list': target_list}


if __name__ == '__main__':

    dataset = MaizeTasselDataset(
        data_dir='dataset',
        data_list='train.txt',
        imgfile="/image/",
        labelfile="/label/",
        train=True,
        transform=transforms.Compose([
            # RandomCrop((256,256),4),
            ToTensor(train=False)]
        )
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )


    valset = MaizeTasselDataset(
        data_dir='dataset',
        data_list='val.txt',
        imgfile="/image/",
        labelfile="/label/",
        train=True,
        transform=transforms.Compose([
            # RandomCrop((256,256),4),
            ToTensor()]
        )
    )

    valloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    print(len(dataloader))
    mean = 0.
    std = 0.
    for i, data in enumerate(dataloader, 0):
        # image_list  = data['image_list']
        images, targets = data['image'], data['target']
        # length = len(image_list)
        # for images in image_list:

        bs = images.size(0)
        images = images.view(bs, images.size(1), -1).float()
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        print(images.size())
        print(i)

    mean /= len(dataloader)*16
    std /= len(dataloader)*16
    print(mean / 255.)
    print(std / 255.)