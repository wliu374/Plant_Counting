import os
import argparse
from time import time
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
# plt.switch_backend('agg')
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import torch.backends.cudnn as cudnn
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks
import torch.optim as optim
from error import *
from IntegrateNet import *
from Netdataset import *

# system-related parameters
data_dir= "dataset"
train_list="train.txt"
val_list="val.txt"

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
optimizer='sgd'
batch_size= 8
crop_size=(256,256)
learning_rate=0.01
# milestones=[200,500]
momentum=0.95
mult=1
num_epoch=1000
weight_decay=0.0005
mae_max = 10000

def save_checkpoint(state, snapshot_dir, filename='model_ckpt.pth.tar'):
    torch.save(state, '{}/{}'.format(snapshot_dir, filename))

# training part
def train(net,train_loader,optimizer,criterion,criterion2,epoch,lamda):
    net.train()  # set train module
    target_filter = torch.cuda.FloatTensor(1, 1, 64, 64).fill_(1) # generate ground truth local count map
    running_loss = 0.0

    for i, sample in enumerate(train_loader):
        inputs, targets = sample['image'], sample['target']
        length = len(inputs) # equal to crop number
        for j in range(length):
            input,target = inputs[j],targets[j]
            input,target= input.cuda(), target.cuda()

            optimizer.zero_grad()
            dic = net(input)
            density = dic['density'] # predicted density map
            local_count = dic['local_count'] # predicted local count map

            loss1 = criterion(density, target)
            target = F.conv2d(target, target_filter, stride=8) # ground truth local count map generated by convolution with density map
            loss2 = criterion2(local_count, target)
            loss = (1-lamda)*loss1+lamda*loss2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    if epoch >900:
            print('epoch: %d, train: %d/%d, ''loss: %.5f' % (epoch,i + 1,len(train_loader),running_loss / (i + 1)))
    net.train_loss['epoch_loss'].append(running_loss / (length*(i + 1))) # record training loss

# testing part
def validate(net, val_loader, epoch,criterion,criterion2,lamda):
    # set evaluation module
    net.eval()

    pd_counts = []
    gt_counts = []
    target_filter = torch.cuda.FloatTensor(1, 1, 64, 64).fill_(1)
    running_loss = 0.0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image, gtcount ,target = sample['image'], sample['gtcount'],sample['target']
            input = image.cuda()
            target = target.cuda()

            dic = net(input)
            output = dic['density'] # predicted density map
            local_count = dic['local_count'] # predicted local count map

            loss1 = criterion(output, target)
            d = F.conv2d(target, target_filter, stride=8) # ground truth local count map
            loss2 = criterion2(local_count, d)

            loss = (1-lamda)*loss1+lamda*loss2
            running_loss += loss.item()
            output = output.squeeze().cpu().detach().numpy()
            output = np.clip(output,0,None) # eliminate < 0 values
            pdcount = output.sum()

            pd_counts.append(pdcount)
            gt_counts.append(gtcount)

    r2 = rsquared(pd_counts, gt_counts)
    mae = compute_mae(pd_counts, gt_counts)
    rmse = compute_rmse(pd_counts, gt_counts)
    print('epoch: {0}, mae: {1:.2f}, rmse: {2:.2f}%, r2: {3:.4f}'.format(epoch, mae, rmse, r2))

    # save stats
    net.val_loss['running_loss'].append(running_loss/(i+1))
    net.val_loss['epoch_loss'].append(mae)
    net.measure['mae'].append(mae)
    net.measure['rmse'].append(rmse)
    net.measure['r2'].append(r2)
    return pd_counts,gt_counts

def main(lamda):
    ## random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(30)
    torch.manual_seed(30)
    np.random.seed(30)

    ## data preprocessing
    train_transforms = transforms.Compose([
        RandomCrop(crop_size, crop_num),
        RandomFlip(),
        Normalize(scale=image_scale, std=image_std, mean=image_mean, train=True),
        ToTensor(train=True),
        ZeroPadding(output_stride, train=True)
    ])

    val_transforms = transforms.Compose([
        Normalize(scale=image_scale, std=image_std, mean=image_mean, train=False),
        ToTensor(train=False),
        ZeroPadding(output_stride, train=False)
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
        num_workers=0,
        pin_memory=True
    )

    # Initiate network
    net = IntegrateNet()
    # net = nn.DataParallel(net)
    net = net.cuda()
    criterion = nn.MSELoss().cuda()
    criterion2 = nn.L1Loss().cuda()

    best_pdcounts = []
    best_gtcounts = []

    # restore parameters
    net.train_loss = {'running_loss': [], 'epoch_loss': []}
    net.val_loss = {'running_loss': [], 'epoch_loss': []}
    net.measure = {'mae': [], 'mse': [], 'rmae': [], 'rmse': [], 'r2': []}

    # define optimizer SGD
    learning_params = [p[1] for p in net.named_parameters()]
    pretrained_params = []
    optimizer = torch.optim.SGD(
        [
            {'params': learning_params},
            {'params': pretrained_params, 'lr': learning_rate / mult},
        ],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_mae = 100000
    best_rmse = 0
    best_r2 = 0

    # milestones=[200,500]
    # start_epoch = 0
    # resume_epoch = -1 if start_epoch == 0 else start_epoch
    # scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=resume_epoch)
    for epoch in range(num_epoch):
        print("epoch", epoch)
        train(net, train_loader, optimizer, criterion, criterion2, epoch, lamda)
        pd_counts, gt_counts = validate(net, val_loader, epoch, criterion, criterion2, lamda)

        # save model parameters
        state = {
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_loss': net.train_loss,
            'val_loss': net.val_loss,
            'measure': net.measure
        }

        # save model parameters each 100 epochs
        if epoch % 100 ==99:
            save_checkpoint(state,"results/models/IntegrateNet",filename='model_ckpt'+str(epoch)+'.pth.tar')

        # save best model parameters and results
        if net.measure['mae'][-1] <= best_mae:
            save_checkpoint(state, "results/models/IntegrateNet", filename='model_best.pth')

            best_mae = net.measure['mae'][-1]
            best_rmse = net.measure['rmse'][-1]
            best_r2 = net.measure['r2'][-1]
            best_pdcounts = pd_counts
            best_gtcounts = gt_counts
        # scheduler.step()

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(net.train_loss['epoch_loss'], label='train loss', color='tab:blue')
    ax1.plot(net.val_loss['running_loss'], label='val loss', color='tab:red')
    ax1.legend(loc='upper right')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(net.val_loss['epoch_loss'], label='val mae', color='tab:orange')
    ax2.legend(loc='upper right')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(net.measure['rmse'], label='val rmse', color='tab:red')
    ax3.legend(loc='upper right')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(net.measure['r2'], label='val r2', color='tab:red')
    ax4.legend(loc='upper right')
    plt.show()

    plt.scatter(best_gtcounts, best_pdcounts)
    plt.plot(best_gtcounts, best_gtcounts)
    plt.xlabel("predicted count of crops", fontsize=15)
    plt.ylabel("ground truth count of crops", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=16)
    plt.show()

    idx = net.measure['mae'].index(min(net.measure['mae']))
    print('lambda',lamda)
    print("the best result is: mae:", best_mae,  'rmse', best_rmse, 'r2',best_r2, "epoch", idx)

if __name__ == "__main__":
    main(0.5)


