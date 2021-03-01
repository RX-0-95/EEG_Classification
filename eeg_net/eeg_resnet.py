from os import scandir
import numpy as np
import torch
from torch._C import MobileOptimizerType, device 
import torch.nn as nn
from torch.types import Device 
from torch.utils.data import Dataset, DataLoader,TensorDataset, random_split 
from torchvision import transforms, utils 
import torch.optim as optim 
import time 
from livelossplot import PlotLosses 
import matplotlib.pyplot as plt 
from eeg_net.utils import *
from eeg_net.eeg_net_base import * 
"""
https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
"""

class Conv2dAuto(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation = 'relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels 
        self.blocks = nn.Identity() 
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity() 
    
    def forward(self,x):
        residual = x 
        if self.should_apply_short: 
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual 
        x = self.activate(x)
        return x 
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels 


class EEGResConv(nn.Module):
    def __init__(self):
        super().__init__()
        



