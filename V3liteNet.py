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
from torch.utils.data import DataLoader

from mixnet import *
from V3litedataset import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder=MixNet(net_type='mixnet_l')
    def forward(self,x):
        x=self.encoder(x)
        return x

class Counter(nn.Module):
    def __init__(self):
        super(Counter, self).__init__()
        self.pool=nn.AvgPool2d(kernel_size=8,stride=1)
        self.conv1=nn.Conv2d(in_channels=56,out_channels=56, kernel_size=(1,1),stride=(1,1))
        self.conv2=nn.Conv2d(in_channels=56,out_channels=1,kernel_size=(1,1),stride=(1,1))
        self.bn1=nn.BatchNorm2d(56)
        self.bn2=nn.BatchNorm2d(1)
        # self.conv1 = nn.Conv2d(in_channels=56, out_channels=128, kernel_size=(8, 8), stride=(1,1), padding=(0,0))
        # self.bn1=nn.BatchNorm2d(num_features=128)
        # self.conv2= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1,1), padding=(0,0))
        # self.conv3=nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1, 1), stride=(1,1), padding=(0,0))

    def forward(self,x):
        x=self.pool(x)
        x=F.relu(self.bn1(self.conv1(x)),inplace=True)
        x=F.relu(self.bn2(self.conv2(x)),inplace=True)
        # x=self.conv3(x)
        return x

# def R_generator(D, p, o, Io):
#     # Input:
#     # D: density map
#     # p: patch size
#     # o: output stride
#     # Io: full-one matrix of size oxo
#     # Output:
#     # R_hat: fine-resolution ground-truth map
#     h, w = D.size()[2:]
#     rh, rw, k = int(h / o), int(w / o), int(p / o)
#     R_hat = F.conv2d(D, Io, stride=o)
#     R_hat = F.unfold(R_hat, kernel_size=k)
#     R_hat = F.fold(R_hat, (rh, rw), kernel_size=k)
#     return R_hat

# class Normalizer:
#     @staticmethod
#     def gpu_normalizer(x, imh, imw, insz, os):
#         _, _, h, w = x.size()
#         accm = torch.cuda.FloatTensor(1, insz*insz, h*w).fill_(1)
#         accm = F.fold(accm, (imh, imw), kernel_size=insz, stride=os)
#         accm = 1 / accm
#         accm /= insz**2
#         accm = F.unfold(accm, kernel_size=insz, stride=os).sum(1).view(1, 1, h, w)
#         x *= accm
#         return x.squeeze().cpu().detach().numpy()

#class Normalizer:
#    @staticmethod
 #   def gpu_normalizer(x):
  #      _, _, rh, rw = x.size()
   #     normalize_ones = torch.cuda.FloatTensor(1, 1, rh, rw).fill_(0)
    #    filter = torch.cuda.FloatTensor(1, 1, 8, 8).fill_(1)
     #   for i in range(rh-7):
      #      for j in range(rw-7):
       #         normalize_ones[0][0][i:i+8,j:j+8]+=filter[0][0]
        #x = x / normalize_ones
        #return x.squeeze().cpu().detach().numpy()

#class dynamic_unfolding(nn.Module):
#    def __init__(self):
#        super(dynamic_unfolding, self).__init__()
#        pass

 #   def forward(self,x,local_count,output_stride):
  #      # print(x)
 #       # print(x.size())
   #     a,b,h,w=x.size()
   #     R=torch.zeros((a,1,h,w)).cuda()
    #    # harray=np.arange(0,h,output_stride)
        # warray=np.arange(0,w,output_stride)
        # print(R.size())
     #3   for batch in range(a):
      #      for i in range(h-output_stride+1):
      #          for j in range(w-output_stride+1):
       #             r=torch.mean(x,dim=1)
       #             r=r[batch][i:i+output_stride,j:j+output_stride]
        #            exp = torch.exp(r)
        #            s = torch.sum(exp)
         #           r = exp / s
         #           R[batch][0][i:i+output_stride,j:j+output_stride]+=local_count[batch][0][i][j]*r
                    # R.unsqueeze(0).unsqueeze(0)
        #return R
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
class V3lite(nn.Module):
    def __init__(self, input_size=64, output_stride=8):
        super(V3lite, self).__init__()
        self.counter=Counter()
        self.encoder = Encoder()
        self.dynamic_unfolding=dynamic_unfolding()
        self.normalizer = Normalizer.gpu_normalizer
        self.weight_init()

    def forward(self,x,is_normalize=False):
        imh, imw = x.size()[2:]
        x = self.encoder(x)
        C = self.counter(x)
        R=self.dynamic_unfolding(local_count=C, output_stride=8,x=x)
        if is_normalize==True:
            R=self.normalizer(R)
        return {'C':C,'R':R}

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

if __name__=='__main__':
    A = torch.randn(8, 3, 256, 256).cuda()
    net = V3lite().cuda()
    net.train()
    dicc = net(A)
    C = dicc["C"]
    R = dicc["R"]
    s = dicc["segmentation"]
    print("end")
    print("C.size", C.size())
    print("R.size", R.size())
    print("s.size", s.size())

