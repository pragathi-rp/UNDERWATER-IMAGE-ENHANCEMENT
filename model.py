import torch
import numpy as np
import torch.nn as nn

class AConvBlock(nn.Module):
    def __init__(self):
        super(AConvBlock,self).__init__()

        block = [nn.Conv2d(3, 64, 3, padding=1), nn.PReLU(), nn.BatchNorm2d(64)]
        block += [nn.Conv2d(64, 64, 3, padding=1), nn.PReLU(), nn.BatchNorm2d(64)]
        block += [nn.AdaptiveAvgPool2d((1,1))]
        block += [nn.Conv2d(64, 32, 1), nn.PReLU(), nn.BatchNorm2d(32)]
        block += [nn.Conv2d(32, 32, 1), nn.PReLU(), nn.BatchNorm2d(32)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class tConvBlock(nn.Module):
    def __init__(self):
        super(tConvBlock,self).__init__()

        block = [nn.Conv2d(6, 8, 3, padding=1), nn.PReLU(), nn.BatchNorm2d(8)]
        block += [nn.Conv2d(8, 8, 3, padding=2, dilation=2), nn.PReLU(), nn.BatchNorm2d(8)]
        block += [nn.Conv2d(8, 8, 3, padding=5, dilation=5), nn.PReLU(), nn.BatchNorm2d(8)]
        block += [nn.Conv2d(8, 3, 3, padding=1), nn.PReLU(), nn.BatchNorm2d(3)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class PhysicalNN(nn.Module):
    def __init__(self):
        super(PhysicalNN,self).__init__()

        self.ANet = AConvBlock()
        self.tNet = tConvBlock()

    def forward(self,x):
        A = self.ANet(x)
        t = self.tNet(torch.cat((x*0+A,x),1))
        out = ((x-A)*t + A)
        return torch.clamp(out, 0., 1.)
