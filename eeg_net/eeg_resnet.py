from os import scandir
import numpy as np
import torch
from torch._C import MobileOptimizerType, device, strided 
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.types import Device 
from torch.utils.data import Dataset, DataLoader,TensorDataset, random_split 
from torchvision import transforms, utils 
import torch.optim as optim 
import time 
from livelossplot import PlotLosses 
import matplotlib.pyplot as plt 
from functools import partial
from eeg_net.utils import *
from eeg_net.eeg_net_base import * 
"""
https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
"""

class Conv2dAuto(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)

conv3x3 = partial(Conv2dAuto,kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation = 'relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels 
        self.activation = activation
        self.blocks = nn.Identity() 
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity() 
     
    def forward(self,x):
        residual = x 
        if self.should_apply_shortcut: 
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual 
        x = self.activate(x)
        return x 
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels 

##Build block enable expanded channels 
class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, 
                    expansion=1,downsampling=1,
                    conv=conv3x3,*args,**kwargs):
        super().__init__(in_channels, out_channels,*args,**kwargs)
        self.expansion, self.downsampling,self.conv = expansion,downsampling,conv 
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels,self.expanded_channels,kernel_size=1,
                stride=self.downsampling,bias=False),
            nn.BatchNorm2d(self.expanded_channels)
        ) if self.should_apply_shortcut else None 

    @property
    def expanded_channels(self):
        return self.out_channels*self.expansion

    @property 
    def should_apply_shortcut(self):
        return self.in_channels!=self.expanded_channels

def conv_bn(in_channels, out_channels, conv,*args,**kwargs):
    return nn.Sequential(
        conv(in_channels,out_channels,*args,**kwargs),
        nn.BatchNorm2d(out_channels)
        )

class ResNetBaiscBlock(ResNetResidualBlock):
    '''
    2 layers of 3x3 conv2d/batchnorm/conv and residual 
    '''
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels,self.out_channels,
                conv=self.conv,bias =False,strided=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels,self.expanded_channels,
                conv=self.conv,bias =False)
        )



class EEGResConv(nn.Module):
    def __init__(self):
        super().__init__()
        



