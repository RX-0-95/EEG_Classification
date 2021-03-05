from os import scandir
import numpy as np
import torch
#from torch._C import TreeView
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


class EEGCNN(nn.Module):
    def __init__(self,in_channels=1, classes=4, *args, **kwargs):
        super().__init__()
        ## Unpack **kwargs 
        _activation = kwargs.pop('activation','none')
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
            out_channels=25, kernel_size=(1,25))
        self.act1 = activation_func(_activation)
        self.conv2 =  nn.Conv2d(in_channels=25,out_channels=25,
            kernel_size=(5,1),stride=(3,1))
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,3)) 

        self.conv3 = nn.Conv2d(in_channels=25,out_channels=50,
            kernel_size=(1,25))
        self.act2 = activation_func(_activation)
        
        self.conv4 = nn.Conv2d(in_channels=50,out_channels=70,
            kernel_size=(3,15),stride=(3,1))
        self.act3 = activation_func(_activation)
        self.conv5 = nn.Conv2d(in_channels=70,out_channels=70,
            kernel_size=(2,1))
        self.conv6 = nn.Conv2d(in_channels=70, out_channels=150,
            kernel_size=(1,15))
        self.linear = nn.Linear(in_features=70*31, out_features=classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = x.view(-1,1,22,1000)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.act2(x)
        x = self.conv4(x)
        x = self.act3(x)
        x = self.maxpool1(x)
        x = self.conv5(x)
        x = self.maxpool1(x)
        x = x.reshape(-1,70*31)
        x = self.linear(x)
        x = self.softmax(x)
        #x = self.conv6(x)
        #x = self.maxpool1(x)
        #x = x.reshape(-1,150*5)
        #x = self.linear(x)
        #x = self.softmax(x)
        return x         


class EEGCNNc2(nn.Module):
    def __init__(self,in_channels=1, classes=4, *args, **kwargs):
        super().__init__()
        ## Unpack **kwargs 
        _activation = kwargs.pop('activation','none')
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
            out_channels=50, kernel_size=(1,25))
        self.act1 = activation_func(_activation)
        self.conv2 =  nn.Conv2d(in_channels=50,out_channels=50,
            kernel_size=(3,1),stride=(2,1))
        self.batchnorm1 = nn.BatchNorm2d(50)
    
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,3)) 
        self.conv3 = nn.Conv2d(in_channels=50,out_channels=80,
            kernel_size=(1,25))
        self.batchnorm2 = nn.BatchNorm2d(80)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)) 
        self.avgpool1 = nn.AvgPool2d((45,1), stride=(7,1))
        self.linear1 = nn.Linear(in_features=800,out_features=40)
        self.linear2 = nn.Linear(in_features=62*40, out_features=classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = x.view(-1,1,22,1000)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x= self.batchnorm2(x)
        x = self.act1(x)
        x = self.maxpool2(x)
        x = x.permute(0,3,2,1)
        xshape = x.shape
        x = x.reshape(xshape[0],xshape[1],-1)
        x = self.linear1(x)
        x = self.act1(x)
        x = torch.square(x)
        x = self.avgpool1(x)
        x = torch.log(x)
        xshape = x.shape
        x = x.reshape(xshape[0],-1)
        x = self.linear2(x)
        x = self.softmax(x) 
      
        return x         





class EEGCNNc3(nn.Module):
    def __init__(self,in_channels=1, classes=4, *args, **kwargs):
        super().__init__()
        ## Unpack **kwargs 
        _activation = kwargs.pop('activation','none')
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
            out_channels=40, kernel_size=(1,25))
        self.act1 = activation_func(_activation)
        self.conv2 =  nn.Conv2d(in_channels=40,out_channels=40,
            kernel_size=(3,1),stride=(2,1))
        self.batchnorm1 = nn.BatchNorm2d(40)
    
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,3)) 
        self.conv3 = nn.Conv2d(kernel_size=(5,5),in_channels=1,out_channels=32)
        self.avgpool3 = nn.AvgPool2d(kernel_size=(5,5),stride=(5,5))
        
        self.conv4 = nn.Conv2d(kernel_size=(3,3),in_channels=32,out_channels=64)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(kernel_size=(1,1),in_channels=64,out_channels=32)

        self.batchnorm2 = nn.BatchNorm2d(40)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)) 
        self.avgpool1 = nn.AvgPool2d((45,1), stride=(7,1))
        self.linear1 = nn.Linear(in_features=32*18*15,out_features=4)
        
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = x.view(-1,1,22,1000)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.maxpool2(x)
 
        x = x.permute(0,3,2,1)
        xshape = x.shape
        x = x.reshape(xshape[0],xshape[1],-1)
        xshape = x.shape
        x = x.view(-1,1,xshape[1],xshape[2])
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act1(x)
        x = self.avgpool3(x)
        x =self.conv4(x)
        x = self.batchnorm4(x)
        x = self.act1(x)
        x = self.avgpool3(x)
        x = self.conv5(x)
        xshape = x.shape
        x = x.reshape(xshape[0],-1)
        x = self.linear1(x)
        x = self.softmax(x)
        
        return x         

