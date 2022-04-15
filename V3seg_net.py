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
import torchvision.transforms.functional as TF
from ASPP import *
from CARAFE import *
from mixnetL import *
from torch.utils.data import DataLoader
from V3segdataset import *

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = MixNet()
        self.aspp = ASPP(264, [6, 12, 18], out_channels=64)
        self.carafe_up = CARAFE_upsampling(64,64)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=(1,1), dilation=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=56, out_channels=64,kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.conv4 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.bn = nn.BatchNorm2d(64)

    def forward(self,x):
        dic = self.encoder(x)
        counting_map = dic['one_8']
        # print('cm',counting_map.size())
        one_8 = self.conv3(counting_map) #convert channel dimension 56 to 64
        one_16 = dic['one_16']
        one_16 = self.conv4(one_16) #convert channel dimension 160 to 64
        one_32 = dic['one_32']

        output = self.aspp(one_32)
        # print(output.size())
        output = self.carafe_up(output)
        if output.size != one_16.size():
            output = TF.resize(output, size=one_16.size()[2:])
        # print(output.size())
        # print(one_16.size())
        output += one_16
        output = self.carafe_up(output)
        if output.size != one_8.size():
            output = TF.resize(output, size=one_8.size()[2:])
        # print(output.size())
        # print(one_8.size())
        output += one_8
        output = F.relu(self.bn(self.conv2(output)),inplace=True)
        dic = {"counting branch": one_8, "segmentation branch": output}

        return dic

class segmenter(nn.Module):
    def __init__(self):
        super(segmenter, self).__init__()
        self.conv=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(1,1),stride=(1,1))
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class Counter(nn.Module):
    def __init__(self):
        super(Counter, self).__init__()
        self.pool=nn.AvgPool2d(kernel_size=8,stride=1)
        self.conv1=nn.Conv2d(in_channels=64,out_channels=64, kernel_size=(1,1),stride=(1,1))
        self.conv2=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(1,1),stride=(1,1))
        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(1)

    def forward(self,x):
        x=self.pool(x)
        x=F.relu(self.bn1(self.conv1(x)),inplace=True)
        x=F.relu(self.bn2(self.conv2(x)),inplace=True)
        # x=self.conv3(x)
        return x

class Normalizer:
    @staticmethod
    def gpu_normalizer(x):
        _, _, rh, rw = x.size()
        normalize_ones = torch.ones((1,1,rh,rw)).cuda()
        normalize_ones = F.unfold(normalize_ones,kernel_size=8)
        normalize_ones = F.fold(normalize_ones,(rh,rw),kernel_size=8)
        x = x/normalize_ones

        return x.squeeze().cpu().detach().numpy()
#
class dynamic_unfolding(nn.Module):
    def __init__(self):
        super(dynamic_unfolding, self).__init__()
        pass

    def forward(self,x,local_count,output_stride):
        # print(x)
        # print(x.size())
        conv_filter = torch.FloatTensor(1,1,8,8).fill_(1).cuda()
        a,b,h,w = x.size()
        avg = torch.mean(x,dim=1,keepdim=True)
        sm = torch.exp(avg)
        sc = F.conv2d(sm,conv_filter,stride=1)
        ssc = sc.reshape((a,-1))
        ssc = torch.unsqueeze(ssc,dim=1)
        ssc = torch.tile(ssc, (1, 64,1))
        uf = F.unfold(sm,kernel_size=8)
        c = local_count.reshape((a,-1))
        c = torch.unsqueeze(c,1)
        c = torch.tile(c, (1, 64,1))
        R = uf/ssc
        R = R*c
        R = F.fold(R,(h,w),kernel_size=8)

        return R

class V3seg(nn.Module):
    def __init__(self):
        super(V3seg, self).__init__()
        self.counter=Counter()
        self.encoder_decoder = EncoderDecoder()
        self.dynamic_unfolding=dynamic_unfolding()
        self.segmenter = segmenter()
        self.normalizer = Normalizer.gpu_normalizer
        self.weight_init()

    def forward(self,x,is_normalize=False):
        dic = self.encoder_decoder(x)
        counting_branch = dic['counting branch']
        softmask = dic['segmentation branch']

        segmentation_branch = self.segmenter(softmask)
        # print(segmentation_branch.size())
        # print("segmentatino_branch",segmentation_branch.size())
        featuremap = counting_branch * segmentation_branch
        # print(counting_branch.size())
        # print(segmentation_branch.size())
        C = self.counter(featuremap)
        R = self.dynamic_unfolding(local_count=C, output_stride=8,x=featuremap)
        if is_normalize==True:
            R=self.normalizer(R)

        # return C,R,segmentation_branch
        return {'C': C, 'R': R, 'segmentation':segmentation_branch}

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight,
                #         mode='fan_in',
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
        A = torch.randn(8,3,256,256).cuda()
        net = V3seg().cuda()
        net.train()
        dicc = net(A)
        C = dicc["C"]
        R = dicc["R"]
        s = dicc["segmentation"]
        print("end")
        print("C.size", C.size())
        print("R.size", R.size())
        print("s.size", s.size())





