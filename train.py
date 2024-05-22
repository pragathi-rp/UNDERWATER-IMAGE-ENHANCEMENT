import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import shutil
import sys
from model import PhysicalNN
from uwcc import UWCCDataset
from action_set import *
from PIL import Image, ImageEnhance
import skimage.color as color
import math
import random
from get_hist import *
from get_global_feature import get_global_feature
from action_set import *
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not torch.multiprocessing.get_start_method(allow_none=True):
    torch.multiprocessing.set_start_method("spawn")

action_size = 20  # Updated action size to include all actions from both snippets

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def take_action(image_np, action_idx):
    return_np = None

    if action_idx == 0:
        return_np = Gamma_up(image_np + 0.5)
    elif action_idx == 1:
        return_np = Gamma_down(image_np + 0.5)
    elif action_idx == 2:
        return_np = contrast(image_np + 0.5, 0.95)
    elif action_idx == 3:
        return_np = contrast(image_np + 0.5, 1.05)
    elif action_idx == 4:
        return_np = color_saturation(image_np + 0.5, 0.95)
    elif action_idx == 5:
        return_np = color_saturation(image_np + 0.5, 1.05)
    elif action_idx == 6:
        return_np = brightness(image_np + 0.5, 0.95)
    elif action_idx == 7:
        return_np = brightness(image_np + 0.5, 1.05)
    elif action_idx == 8:
        r, g, b = 245, 255, 255  # around 6300K
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 9:
        r, g, b = 265, 255, 255  # around 6300K
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 10:
        r, g, b = 255, 245, 255  # around 6300K
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 11:
        r, g, b = 255, 265, 255  # around 6300K
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 12:
        r, g, b = 255, 255, 245  # around 6300K
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 13:
        r, g, b = 255, 255, 265  # around 6300K
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 14:
        return_np = HE(image_np + 0.5)
    elif action_idx == 15:
        return_np = CLAHE(image_np + 0.5)
    elif action_idx == 16:
        return_np = white_balance(image_np + 0.5, 0.5)
    elif action_idx == 17:
        return_np = sharpen(image_np + 0.5)
    elif action_idx == 18:
        return_np = emboss(image_np + 0.5)
    elif action_idx == 19:
        return_np = DCP(image_np + 0.5)
    else:
        print("error")
    return return_np - 0.5

def main():
    best_loss = float('inf')

    lr = 0.001
    batch_size = 16
    num_workers = 2
    epochs = 50
    ori_fd = sys.argv[1]
    ucc_fd = sys.argv[2]
    ori_dirs = [os.path.join(ori_fd, f) for f in os.listdir(ori_fd)]
    ucc_dirs = [os.path.join(ucc_fd, f) for f in os.listdir(ucc_fd)]

    print("Original directories:", ori_dirs)
    print("Corrected directories:", ucc_dirs)

    if len(ori_dirs) == 0 or len(ucc_dirs) == 0:
        raise RuntimeError('Found 0 image pairs in given directories.')

    model = PhysicalNN()
    model = nn.DataParallel(model)
    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    trainset = UWCCDataset(ori_dirs, ucc_dirs, train=True)
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers)

    for epoch in range(epochs):

        tloss = train(trainloader, model, optimizer, criterion, epoch)

        print('Epoch:[{}/{}] Loss{}'.format(epoch, epochs, tloss))
        is_best = tloss < best_loss
        best_loss = min(tloss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
    print('Best Loss: ', best_loss)

def train(trainloader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    model.train()

    for i, sample in enumerate(trainloader):
        ori, ucc = sample
        ori = ori.to(device)
        ucc = ucc.to(device)

        ori = torch.nn.functional.interpolate(ori, size=(480, 640), mode='bilinear', align_corners=False)
        ucc = torch.nn.functional.interpolate(ucc, size=(480, 640), mode='bilinear', align_corners=False)

        corrected = model(ori)
        loss = criterion(corrected, ucc)
        losses.update(loss.item(), ori.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg

def save_checkpoint(state, is_best):
    freq = 500
    epoch = state['epoch']

    filename = './checkpoints/model_tmp.pth.tar'
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    torch.save(state, filename)

    if epoch % freq == 0:
        shutil.copyfile(filename, './checkpoints/model_{}.pth.tar'.format(epoch))
    if is_best:
        shutil.copyfile(filename, './checkpoints/model_best_{}.pth.tar'.format(epoch))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
