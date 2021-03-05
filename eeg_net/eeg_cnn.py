from os import scandir
import numpy as np
import torch

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


def conv_bn(in_channels, out_channels, conv,*args,**kwargs):
    return nn.Sequential(
        conv(in_channels,out_channels,*args,**kwargs),
        nn.BatchNorm2d(out_channels)
        )

def pool_fn(pool_type,kernel_size,stride):
    return nn.ModuleDict([
        ['max',nn.MaxPool2d(kernel_size=kernel_size,stride=stride)],
        ['avg',nn.AvgPool2d(kernel_size=kernel_size,stride=stride)]
    ])[pool_type]


class EEGCNNv2Encoder(nn.Module):
    def __init__(self,in_channels,input_size=(1,22,1000),options={},*args,**kwargs):
        super().__init__()
        self.in_size = input_size
        ## Unpack varaibles 
        _gate_conv_size = options.pop('gate_conv_size',25)
        _gate_conv_out_channel = options.pop('gate_conv_out_channel',50)
        _prob_conv_size = options.pop('prob_conv_size',3)
        _prob_conv_downsample = options.pop('prob_down_sample',2)
        _feature_conv_size = options.pop('feature_conv_size',25)
        _feature_conv_out_channel = options.pop('feature_conv_out_channel',80)
        _feature_pool_type = options.pop('feature_pool_type','max')
        _feature_pool_size = options.pop('feature_pool_size',2)
        _activation = options.pop('activation','none')

        if len(options) >0:
            extra = ', '.join('"%s"' % k for k in list(options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra)

        self.gate_conv = conv_bn(conv=nn.Conv2d,kernel_size=(1,_gate_conv_size),
            in_channels=in_channels,out_channels=_gate_conv_out_channel)
        self.prob_conv = conv_bn(conv=nn.Conv2d,kernel_size=(_prob_conv_size,1),
            in_channels=_gate_conv_out_channel,out_channels=_gate_conv_out_channel)
        self.activation = activation_func(_activation)
        self.feature_conv =  conv_bn(conv=nn.Conv2d,kernel_size=(1,_feature_conv_size),
            in_channels=_gate_conv_out_channel,out_channels=_feature_conv_out_channel)
        #self.feature_pool = pool_fn('')
        self.pool1 = pool_fn(_feature_pool_type,kernel_size=(1,_feature_pool_size),stride=(1,_feature_pool_size))
        self.blocks = nn.Sequential(
            conv_bn(conv=nn.Conv2d,kernel_size=(1,_gate_conv_size),
            in_channels=in_channels,out_channels=_gate_conv_out_channel),
            activation_func(_activation),
            conv_bn(conv=nn.Conv2d,kernel_size=(_prob_conv_size,1),stride=(_prob_conv_downsample,1),
            in_channels=_gate_conv_out_channel,out_channels=_gate_conv_out_channel),
            activation_func(_activation),
            conv_bn(conv=nn.Conv2d,kernel_size=(1,_feature_conv_size),
            in_channels=_gate_conv_out_channel,out_channels=_feature_conv_out_channel),
            activation_func(_activation),
            pool_fn(_feature_pool_type,kernel_size=(1,_feature_pool_size),stride=(1,_feature_pool_size))
        )

    def forward(self,x):
        x = self.blocks(x)
        return x 

    @property
    def output_size(self):
        dummy =  torch.rand(size=self.in_size)
        out = self.forward(dummy.unsqueeze(0).to(next(self.parameters()).device))
        return out.shape 
        
        
class EEGCNNv2Decoder(nn.Module):
    def __init__(self,in_size, classes,options={},*args,**kwargs):
        super().__init__()
        ## Unpack options 
        _linear1_out = options.pop('linear1_out',40)
        _activation = options.pop('activation','none')
        _pool_type = options.pop('pool_type','avg')
        _pool_size = options.pop('pool_size',45)
        _pool_downsample = options.pop('pool_downsample',7)
        if len(options) >0:
            extra = ', '.join('"%s"' % k for k in list(options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra)
        in_features1 = in_size[1]*in_size[2]
        in_features2 = (in_size[3]-_pool_size)//_pool_downsample + 1 

        self.linear1 = nn.Linear(in_features=in_features1,out_features=_linear1_out)
        self.linear2 = nn.Linear(in_features=in_features2*_linear1_out,out_features=classes)
        self.activation = activation_func(_activation)
        self.pool = pool_fn(_pool_type,kernel_size=(_pool_size,1),stride=(_pool_downsample,1))
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = x.permute(0,3,2,1)
        xshape = x.shape
        x = x.reshape(xshape[0],xshape[1],-1)
        #print(x.shape)
        x = self.linear1(x)
        x = self.activation(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(x)
        xshape = x.shape
        x = x.reshape(xshape[0],-1)
        x = self.linear2(x)
        x = self.softmax(x) 
        return x 

class EEGCNNv2(nn.Module):
    """
    EEGCNNV2: 
    1. Conv1d along the time space of the signal, generate feature map of time interval 
    2. Conv1d along the 22 prob, learn feature of the those probs 
    3. Conv1d along the time space, increase the feature maps 
    4. maxpool along time space shrink the time space length 
    5. flatten the prob space and feature space 
    6. fc along the prob feature space 
    7. fc along time space 
    """
    def __init__(self, in_channels=1, classes=4,input_size=(1,22,1000),encoder_opt={},
        decoder_opt={},*args, **kwargs):
        super().__init__()
        self.input_size = input_size
        ## Unpack **kwargs 
        self.encoder = EEGCNNv2Encoder(in_channels=in_channels,
            options=encoder_opt,input_size=input_size)
        encoder_out_size = self.encoder.output_size
        
        self.decoder = EEGCNNv2Decoder(in_size=encoder_out_size,classes=classes,
            options=decoder_opt)
    def forward(self,x):
        x = x.view(-1,*self.input_size)
        x = self.encoder(x)
        x = self.decoder(x)
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
        
        self.avgpool1 = nn.AvgPool2d((45,1), stride=(7,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,3)) 

        self.avgpool2 = nn.AvgPool2d(kernel_size=(1,2),stride=(1,2)) 
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)) 
        self.avgpool3 = nn.AvgPool2d(kernel_size=(5,5),stride=(5,5))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(5,5),stride=(5,5)) 

        self.pool1 = self.avgpool1
        self.pool2 = self.avgpool2
        self.pool3 = self.avgpool3

        #self.pool1 = self.maxpool1
        #self.pool2 = self.maxpool2
        #self.pool3 = self.maxpool3

        self.conv3 = nn.Conv2d(kernel_size=(5,5),in_channels=1,out_channels=32)

        
        self.conv4 = nn.Conv2d(kernel_size=(3,3),in_channels=32,out_channels=64)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(kernel_size=(1,1),in_channels=64,out_channels=32)

        self.batchnorm2 = nn.BatchNorm2d(40)
        self.batchnorm3 = nn.BatchNorm2d(32)
        
        
        self.linear1 = nn.Linear(in_features=32*18*15,out_features=100)
        self.linear2 = nn.Linear(in_features=100,out_features=4)
        
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
      
        x = self.pool3(x)
       
        x =self.conv4(x)
        x = self.batchnorm4(x)
        x = self.act1(x)
       
        x = torch.square(x)
        x = self.avgpool3(x)
        x = torch.log(x)
        x = self.conv5(x)
        xshape = x.shape
        x = x.reshape(xshape[0],-1)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        
        return x         

