from os import scandir
import numpy as np
import torch
#from torch._C import MobileOptimizerType, device, strided 
import torch.nn as nn
from torch.nn.modules import activation, conv
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.pooling import MaxPool2d
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

class Conv2dNopad(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

class AvgPoolAuto(nn.AvgPool2d):
    pass 

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

class ResNetBasicBlock(ResNetResidualBlock):
    '''
    2 layers of 3x3 conv2d/batchnorm/conv and residual 
    '''
    expansion = 1 
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels,self.out_channels,
                conv=self.conv,bias =False,stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels,self.expanded_channels,
                conv=self.conv,bias =False)
        )
   
class ResNetBottleNeckBlock(ResNetResidualBlock):
    """
    The ResNetBottleNeckBlock is used to keep the parameters size as low 
    as possible. 
    Example: If the BottleNeck is not applied:
    input: (-1,32,10,10) to 
    output: (-1,256,10,10)
    will need 256*32 (3x3) kernel. But if the BottleNeck is applied 
    we will need: 32*64 (1x1) kernel, 64*64 (3x3) kernel, 64*256 (1x1) kernel
    """
    expansion = 4 
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels,self.out_channels,
                self.conv,kernel_size=1),
            activation_func(self.activation),
            conv_bn(self.out_channels,self.out_channels,
                self.conv,kernel_size=3,stride=self.downsampling), 
            activation_func(self.activation),
            conv_bn(self.out_channels,self.expanded_channels,
                self.conv,kernel_size=1)
        )

class ResNetLayer(nn.Module):
    """
    ResNet layer consist n blocks ResNetBasicBlocks
    """
    def __init__(self,in_channels,out_channels,block=ResNetBasicBlock,n=1,*args,**kwargs):
        super().__init__()
        # Downwsampling directly by convolutional layer that have stride of 2 
        downsampling = 2 if in_channels != out_channels else 1 
        self.blocks = nn.Sequential(
            block(in_channels,out_channels,*args,**kwargs,downsampling=downsampling),
            *[block(out_channels*block.expansion,out_channels,downsampling=1,
                *args,**kwargs) for _ in range(n-1)]
        )
    def forward(self,x):
        x =self.blocks(x)
        return x 

class ResNetEncoder(nn.Module):
    def __init__(self,in_channels=3, block_sizes=[64,128,256,512],deepths=[2,2,2,2],
        activaion='relu',block=ResNetBasicBlock,*args,**kwargs):
        super().__init__()
        self.blocks_sizes = block_sizes
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels,self.blocks_sizes[0],
                    kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(self.block_szies[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.in_out_block_sizes = list(zip(block_sizes,block_sizes[1:]))

class EEGResNetV2(nn.Module):
    def __init__(self,in_channels, classes):
        super().__init__()
        """
        1. Conv1d (32,(1,25)): Process time interval 
        * input: (B,1,22,1000)
        * output: (B,32,22,976)
        """
        self.conv1 = nn.Conv2d(in_channels,32,(1,25),stride=1)
        """
        2. Conv1d (32,64,(1,25)): Process time interval 
        * input: (B,32,22,976)
        * output: (B,64,22,952)
        """
        self.conv2 = nn.Conv2d(32,64,(1,25),stride=1)

        """
        ReLU 
        """
        self.Relu = activation_func('leaky_relu')

        """
        4. (*) Max pool(1d ) (1,22,stride = 10)   // may need futuer incrase stride or 
        * input(B,64,22,952)                   //incrase the kernel size in conv2d below
        * output (B,64,22,94)
        """
        self.maxpool1 = nn.MaxPool2d((1,22),stride=(1,10))
        self.avgpool = nn.AvgPool2d((1,22), stride=(1,10))
        """
        5. Residual block1  
        * input(B,64,22,94)                   
        * output (B,64,22,94)
        """
        self.resudual_block1 = ResNetBasicBlock(in_channels=32,out_channels=64,activation='none')
        self.resudual_block2 = ResNetBasicBlock(in_channels=64,out_channels=128,activation='none')
        self.resudual_block3 = ResNetBasicBlock(in_channels=128,out_channels=128,activation='none')
        self.resudual_block4 = ResNetBasicBlock(in_channels=128,out_channels=64,activation='none')
        """
        6. Fullconnect (in:1408, out:32)
        * input(B,64,22,94)
            * reform (B,94,64,22)
            * reform (B,94,64*22)
        * out(B,94,32)
        * square 
        * log 
        """
        self.fc1 = nn.Linear(1408,32)
        
       
        """
        8. fc2 (in: 1048*32, out: 4)
        * input: (B,94,32) 
        * reshape: (B,94*32)
        * fc: out(B,4)
        """       
        self.fc2 = nn.Linear(96*32,4) 
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = x.view(-1,1,22,1000)
        x = self.conv1(x)
        #(x.shape)
        x = self.Relu(x)
        #x = self.conv2(x)
        #x = self.Relu(x)
        #print(x.shape)
        x = torch.square(x)
        x = self.maxpool1(x)
        x = torch.log(x)
        #
        x = self.resudual_block1(x)
        x = self.resudual_block2(x)
        x = self.resudual_block3(x)
        x = self.resudual_block4(x)
        
        
        #print(x.shape)
        #6 
        x = x.permute(0,3,2,1)
        x = x.reshape(-1,96,64*22)
        #print(x.shape)
        x = self.fc1(x)
        #x = torch.square(x)
        x = x.permute(0,2,1)
        #x = torch.log(x)
        x = x.reshape(-1,96*32)
        x = self.fc2(x)
        x = self.softmax(x)
        return x 
        
 
class EEGResNet(nn.Module):
    def __init__(self,in_channels, classes):
        super().__init__()
        """
        1. Conv1d (32,(1,25)): Process time interval 
        * input: (B,1,22,1000)
        * output: (B,32,22,976)
        """
        self.conv1 = nn.Conv2d(in_channels,32,(1,25),stride=1)
        """
        2. Conv1d (32,64,(1,25)): Process time interval 
        * input: (B,32,22,976)
        * output: (B,64,22,952)
        """
        self.conv2 = nn.Conv2d(32,64,(1,25),stride=1)

        """
        ReLU 
        """
        self.Relu = activation_func('leaky_relu')

        """
        4. (*) Max pool(1d ) (1,22,stride = 10)   // may need futuer incrase stride or 
        * input(B,64,22,952)                   //incrase the kernel size in conv2d below
        * output (B,64,22,94)
        """
        self.maxpool1 = nn.MaxPool2d((1,22),stride=(1,10))
        """
        5. Residual block1  
        * input(B,64,22,94)                   
        * output (B,64,22,94)
        """
        self.resudual_block1 = ResNetBasicBlock(in_channels=32,out_channels=64,activation='none')
        self.resudual_block2 = ResNetBasicBlock(in_channels=64,out_channels=64,activation='none')
        """
        6. Fullconnect (in:1408, out:32)
        * input(B,64,22,94)
            * reform (B,94,64,22)
            * reform (B,94,64*22)
        * out(B,94,32)
        * square 
        * log 
        """
        self.fc1 = nn.Linear(1408,32)
        
        self.avgpool = nn.AvgPool1d(75, stride=15)
        """
        8. fc2 (in: 1048*32, out: 4)
        * input: (B,94,32) 
        * reshape: (B,94*32)
        * fc: out(B,4)
        """       
        self.fc2 = nn.Linear(96*32,4) 
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = x.view(-1,1,22,1000)
        x = self.conv1(x)
        #(x.shape)
        x = self.Relu(x)
        #x = self.conv2(x)
        #x = self.Relu(x)
        #print(x.shape)
        x = self.maxpool1(x)
        #
        x = self.resudual_block1(x)
        x = self.resudual_block2(x)
        x = self.resudual_block2(x)
        x = self.resudual_block2(x)
        
        
        #print(x.shape)
        #6 
        x = x.permute(0,3,2,1)
        x = x.reshape(-1,96,64*22)
        #print(x.shape)
        x = self.fc1(x)
        #x = torch.square(x)
        x = x.permute(0,2,1)
        #x = torch.log(x)
        x = x.reshape(-1,96*32)
        x = self.fc2(x)
        x = self.softmax(x)
        return x 
    
    """
    def forward(self,x):
        x = x.view(-1,1,22,1000)
        x = self.conv1(x)
        #(x.shape)
        x = self.Relu(x)
        #x = self.conv2(x)
        #x = self.Relu(x)
        #print(x.shape)
        x = self.maxpool1(x)
        #
        x = self.resudual_block1(x)
        x = self.resudual_block2(x)
        x = self.resudual_block2(x)
        x = self.resudual_block2(x)
        
        
        #print(x.shape)
        #6 
        x = x.permute(0,3,2,1)
        x = x.reshape(-1,96,64*22)
        #print(x.shape)
        x = self.fc1(x)
        x = torch.square(x)
        x = x.permute(0,2,1)

        x = torch.log(x)
        x = x.reshape(-1,96*32)
        x = self.fc2(x)
        x = self.softmax(x)
        return x 
    """


