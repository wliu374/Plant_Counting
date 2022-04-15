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
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader

from error import *
from V3liteNet import *
from V3litedataset import *
# prevent dataloader deadlock, uncomment if deadlock occurs
# cv.setNumThreads(0)

# system-related parameters
data_dir= "dataset"
train_list="train.txt"
val_list="val.txt"

image_scale = 1. / 255
# image_mean = [0.01, 0.01, 0.01]
image_mean = [0.0210, 0.0193, 0.0181]
image_std = [1, 1, 1]
image_mean = np.array(image_mean).reshape((1, 1, 3))
image_std = np.array(image_std).reshape((1, 1, 3))
input_size=64
output_stride=8
resized_ratio=0.125
print_every=1

crop_num = 8
# model-related parameters
optimizer='sgd'
batch_size= 8
crop_size=(256,256)
learning_rate=0.000001
milestones=[200,400,700]
momentum=0.95
mult=1
num_epoch=1000
weight_decay=0.0005
mae_max = 10000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def save_checkpoint(state, snapshot_dir, filename='model_ckpt.pth.tar'):
    torch.save(state, '{}/{}'.format(snapshot_dir, filename))
def R_generator(D, p, o, Io):
    # Input:
    # D: density map
    # p: patch size
    # o: output stride
    # Io: full-one matrix of size oxo
    # Output:
    # R_hat: fine-resolution ground-truth map
    h, w = D.size()[2:]
    rh, rw, k = int(h / o), int(w / o), int(p / o)
    R_hat = F.conv2d(D, Io, stride=o)
    R_hat = F.unfold(R_hat, kernel_size=k)
    R_hat = F.fold(R_hat, (rh, rw), kernel_size=k)
    return R_hat

Io=torch.FloatTensor(1, 1, 8, 8).fill_(1).cuda()

def train(net,train_loader,criterion,optimizer,epoch):
    # switch to train mode
    net.train()
    cudnn.benchmark = True

    if batch_size == 1:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    running_loss=0.0
    in_sz=input_size
    os=output_stride
    target_filter = torch.cuda.FloatTensor(1, 1, in_sz, in_sz).fill_(1)
    for i, sample in enumerate(train_loader):
        image_list, target_list = sample['image_list'], sample['target_list']
        length = len(image_list)

        # zero the parameter gradients
        optimizer.zero_grad()
        for j in range(length):
            input,target = image_list[j], target_list[j]
            inputs, target = input.cuda(), target.cuda()

            # forward
            y=net(inputs)
            C=y["C"]
            R=y["R"]

            R_hat=R_generator(target,64,8,Io)
            # generate targets
            targets = F.conv2d(target, target_filter, stride=os)

            # compute loss
            loss1 = criterion(C, targets)
            loss2 = criterion(R,R_hat)
            loss=loss1+loss2

            # backward + optimize
            loss.backward()
            optimizer.step()

            # collect and print statistics
            running_loss += loss.item()

        if epoch>=0 and i % print_every == print_every-1:
            print('epoch: %d, train: %d/%d, '
                  'loss: %.5f' % (
                      epoch,
                      (i+1)*length,
                      length*len(train_loader),
                      running_loss / (length*(i+1)),

                  ))
    net.train_loss['epoch_loss'].append(running_loss / (length*(i + 1)))


def validate(net,valset,val_loader,criterion,epoch,plot=False):
    net.eval()
    cmap = plt.cm.get_cmap('jet')
    image_list=valset.image_list

    pd_counts = []
    gt_counts = []
    with torch.no_grad():
        idx = np.random.randint(low=0, high=20, size=1)
        for i, sample in enumerate(val_loader):
            image , gtcount ,targets= sample['image'], sample['gtcount'],sample['target']
            # output=net(image.cuda(),is_normalize=True)['R']
            # output=np.clip(output.squeeze().cpu().detach().numpy(), 0, None)
            # pdcount=output.sum()
            # gtcount=float(gtcount.numpy())

            dic=net(image.cuda(), is_normalize=False)
            R = dic['R']
            # print(R.size())
            R=Normalizer.gpu_normalizer(R)
            R=np.clip(R,0,None)
            pdcount=R.sum()
            gtcount=float(gtcount.numpy())
            # output = net(image.cuda(), is_normalize=False)['C']
            # output=output=Normalizer.gpu_normalizer(output,image.size()[2], image.size()[3], input_size, output_stride)
            # output=np.clip(output,0,None)
            # pdcount=output.sum()
            # gtcount=float(gtcount.numpy())

            if plot == True:
                if i==idx:
                    _, image_name = os.path.split(image_list[i])
                    output_save=net(image.cuda(),is_normalize=False)["R"]
                    output_save = np.clip(output_save.squeeze().cpu().numpy(), 0, None)
                    output_save = recover_countmap(output_save, image, input_size, output_stride)
                    output_save = output_save / (output_save.max() + 1e-12)
                    output_save = cmap(output_save) * 255.
                    # image composition
                    image = valset.images[image_list[i]]
                    nh,nw = image.shape[:2]
                    # nh, nw = output_save.shape[:2]
                    # image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
                    output_save = cv2.resize(output_save, (nw, nh), interpolation=cv2.INTER_CUBIC)
                    output_save = 0.5 * image + 0.5 * output_save[:, :, 0:3]
                    dotimage = valset.dotimages[image_list[i]]
                    target_image=valset.targets[image_list[i]]

                    fig = plt.figure()
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax1.imshow(dotimage.astype(np.uint8))
                    ax1.get_xaxis().set_visible(False)
                    ax1.get_yaxis().set_visible(False)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax2.imshow(target_image.squeeze())
                    ax2.get_xaxis().set_visible(False)
                    ax2.get_yaxis().set_visible(False)
                    ax3=fig.add_subplot(1,3,3)
                    ax3.imshow(output_save.astype(np.uint8))
                    ax3.get_xaxis().set_visible(False)
                    ax3.get_yaxis().set_visible(False)
                    fig.suptitle('manual count=%4.2f, inferred count=%4.2f' % (gtcount, pdcount),
                                 fontsize=10)
                    plt.tight_layout(rect=[0, 0, 1, 1.4])
                    plt.show()

            # compute mae and mse
            pd_counts.append(pdcount)
            gt_counts.append(gtcount)
            mae = compute_mae(pd_counts, gt_counts)
            mse = compute_mse(pd_counts, gt_counts)
            rmae, rmse = compute_relerr(pd_counts, gt_counts)

            if epoch>=400 and i % print_every == print_every - 1:
                print(
                    'epoch: {0}, test: {1}/{2}, pre: {3:.2f}, gt:{4:.2f}, me:{5:.2f}, mae: {6:.2f}, mse: {7:.2f}, rmae: {8:.2f}%, rmse: {9:.2f}%, '
                        .format(epoch, i + 1, len(val_loader), pdcount, gtcount, pdcount - gtcount, mae, mse, rmae,
                                rmse)
                )

    # print("gtcounts",gt_counts)
    r2 = rsquared(pd_counts, gt_counts)
    mae = compute_mae(pd_counts, gt_counts)
    rmse = compute_rmse(pd_counts, gt_counts)

    print('epoch: {0}, mae: {1:.2f},rmse: {2:.2f}%, r2: {3:.4f}'.format(epoch, mae, rmse, r2))

    # save stats
    net.val_loss['epoch_loss'].append(mae)
    net.measure['mae'].append(mae)
    net.measure['rmse'].append(rmse)
    net.measure['r2'].append(r2)

    if plot==True:
            plt.scatter(gt_counts,pd_counts)
            plt.plot(gt_counts,gt_counts)
            plt.xlabel("gt_counts")
            plt.ylabel("pd_counts")
            plt.show()

def main():
    # initial network
    if torch.cuda.is_available():
        torch.cuda.manual_seed(10)
    torch.manual_seed(10)
    np.random.seed(10)
    net=V3lite()
    net=nn.DataParallel(net)
    net.cuda()

    # filter parameters
    learning_params = [p[1] for p in net.named_parameters()]
    pretrained_params = []

    # define loss function and optimizer
    criterion = nn.L1Loss(reduction='mean').cuda()

    optimizer=torch.optim.SGD(
            [
                {'params': learning_params},
                {'params': pretrained_params, 'lr': learning_rate/ mult},
            ],
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

    # restore parameters
    start_epoch=0
    net.train_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.val_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.measure = {
        'mae': [],
        'mse': [],
        'rmae': [],
        'rmse': [],
        'r2': []
    }

    # define dataset loader
    train_transforms = transforms.Compose([
        RandomCrop(crop_size,8),
        RandomFlip(),
        Normalize(scale=image_scale, std=image_std, mean=image_mean,train = True),
        ToTensor(),
        ZeroPadding(output_stride)

    ])
    val_transforms = transforms.Compose([
        Normalize(scale=image_scale, std=image_std, mean=image_mean,train=False),
        ToTensor(train = False),
        ZeroPadding(output_stride,train = False)
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
    valset = MaizeTasselDataset(
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

    resume_epoch = -1 if start_epoch == 0 else start_epoch
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=resume_epoch)

    best_mae = 100000000.0
    for epoch in range(num_epoch):
        print("epoch",epoch)
        # train


        train(net,train_loader,criterion,optimizer,epoch)
        if epoch % 100 ==99:
            validate(net,valset,val_loader,criterion,epoch+1,plot=True)
        else:
            validate(net, valset, val_loader, criterion, epoch + 1, plot=False)
        state = {
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_loss': net.train_loss,
            'val_loss': net.val_loss,
            'measure': net.measure
        }
        if epoch % 100 ==99:
            save_checkpoint(state, 'results/v3lite', filename='model_ckpt'+str(epoch)+'.pth.tar')
        if net.measure['mae'][-1] <= best_mae:
            best_mae = net.measure['mae'][-1]
            save_checkpoint(state,'results/v3lite',filename='model_best.pth.tar')
        scheduler.step()

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(net.train_loss['epoch_loss'], label='train loss', color='tab:blue')
    ax1.legend(loc='upper right')
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(net.val_loss['epoch_loss'], label='val mae', color='tab:orange')
    ax2.legend(loc='upper right')
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(net.measure['mse'], label='val mse', color='tab:green')
    ax3.legend(loc='upper right')
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(net.measure['rmae'], label='val rmae', color='tab:red')
    ax4.legend(loc='upper right')
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(net.measure['rmse'], label='val rmse', color='tab:red')
    ax5.legend(loc='upper right')
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(net.measure['r2'], label='val r2', color='tab:red')
    ax6.legend(loc='upper right')
    plt.show()
    idx=net.measure['mae'].index(min(net.measure['mae']))
    mae, mse, rmae, rmse, r2 = net.measure['mae'][idx], net.measure['mse'][idx], net.measure['rmae'][idx],net.measure['rmse'][idx],net.measure['r2'][idx]
    print("the best result is: mae:", mae, 'mse:',mse, 'rmae',rmae,'rmse',rmse,'r2',r2)


if __name__ == "__main__":
    main()


