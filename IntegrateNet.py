from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from sympy import *
import os
import json
import pandas as pd
import random
from PIL import Image
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
from torch.utils.data import DataLoader
from mixnetL import *
from ASPP import *
from CARAFE import *

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Counter(nn.Module):
    def __init__(self):
        super(Counter, self).__init__()
        self.pool=nn.AvgPool2d(kernel_size=64,stride=8)
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),stride=(1,1))
        self.conv2=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(1,1),stride=(1,1))
        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(1)

    def forward(self,x):
        x=self.pool(x)
        x=F.relu(self.bn1(self.conv1(x)),inplace=True)
        x=F.relu(self.bn2(self.conv2(x)),inplace=True)
        return x

class IntegrateNet(nn.Module):
    def __init__(self):
        super(IntegrateNet, self).__init__()
        self.Encoder = MixNet()
        # self.ASPP1 = ASPP(264, [6, 12, 18], out_channels=256)
        self.CARAFE1 = CARAFE_upsampling(256,128)
        self.CARAFE2 = CARAFE_upsampling(128,64)
        self.CARAFE3 = CARAFE_upsampling(64,64,delta=8)
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=56, out_channels=64,kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.Doubleconv1 = DoubleConv(in_channels=256,out_channels=128)
        self.Doubleconv2 = DoubleConv(in_channels=128,out_channels=64)
        self.Doubleconv3 = DoubleConv(in_channels=64,out_channels= 64)
        self.Doubleconv4 = DoubleConv(in_channels=64,out_channels=1)
        self.Doubleconv5 = DoubleConv(in_channels=264,out_channels=256)
        self.Counting = Counter()
        self.weights_init()

    def forward(self,output):
        # upsampling #1
        imgsize = output.size()
        dic = self.Encoder(output)
        one_8,one_16,one_32 = dic['one_8'],dic['one_16'],dic['one_32']
        # output = self.ASPP1(one_32)
        output = self.Doubleconv5(one_32)
        output = self.CARAFE1(output)
        one_16 = self.conv1(one_16)
        if output.size() != one_16.size(): ## avoid different sizes through upsampling
            output = TF.resize(output,size=one_16.size()[2:])
        output = torch.cat((one_16,output),dim=1)
        output = self.Doubleconv1(output)

        # upsampling #2
        output = self.CARAFE2(output)
        one_8 = self.conv2(one_8)
        if output.size() != one_8.size():
            output = TF.resize(output,size=one_8.size()[2:])
        output = torch.cat((one_8,output),dim=1)
        output = self.Doubleconv2(output)

        # upsampling #3
        output = self.CARAFE3(output)
        if output != imgsize:
            output = TF.resize(output,size=imgsize[2:])
        output = self.Doubleconv3(output)
        local_count = self.Counting(output)
        # output = self.Doubleconv3(output)
        output = self.conv3(output)
        return {'density':output,'local_count':local_count}

    ## initial weights
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    net =IntegrateNet().cuda()
    x = net(torch.randn(8,3,320,320).cuda())
    print("OK")
    x = net(x)['density']
    local_count = net(x)['local_count']
    print(x.size())
    print(local_count.size())



































