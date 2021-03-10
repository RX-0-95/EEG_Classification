
from eeg_net.eeg_resnet import Conv1dAuto, Conv2dAuto
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
def conv_nbn(in_channels, out_channels, conv,*args,**kwargs):
    return nn.Sequential(
        conv(in_channels,out_channels,*args,**kwargs)
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


class EEGCNNv3Encoder(nn.Module):
    def __init__(self,in_channels,input_size=(1,22,1000),options={},*args,**kwargs):
        super().__init__()
        self.in_size = input_size
        ## Unpack varaibles 
        _conv1_size = options.pop('conv1_size',3)
        _conv1_out_channel = options.pop('conv1_out_channel',22)
        _conv1_pool = options.pop('conv1_pool',2)
        _conv2_size = options.pop('conv2_size',3)
        _conv2_out_channel = options.pop('conv2_out_channel',44)
        _conv2_pool = options.pop('conv2_pool',2)
        _conv3_size = options.pop('conv3_size',3)
        _conv3_out_channel = options.pop('conv3_out_channel',22)
        _conv3_pool = options.pop('conv3_pool',3)
        _feature_pool_type = options.pop('pool_type','max')
        _activation = options.pop('activation','none')
        conv1 = partial(Conv2dAuto,kernel_size=(1,_conv1_size),stride = (1,1))
        conv2 = partial(Conv2dAuto,kernel_size=(1,_conv2_size),stride = (1,1))
        conv3 = partial(Conv2dAuto,kernel_size=(1,_conv3_size),stride = (1,1))
        #conv1 = partial(Conv1dAuto,kernel_size=_conv1_size,stride = 1)
        #conv2 = partial(Conv1dAuto,kernel_size=_conv2_size,stride = 1)
        #conv3 = partial(Conv1dAuto,kernel_size=_conv3_size,stride = 1)
        
        if len(options) >0:
            extra = ', '.join('"%s"' % k for k in list(options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra)
       
        self.blocks = nn.Sequential(
            conv_bn(conv=conv1,in_channels=in_channels,out_channels=_conv1_out_channel,bias=False),
            pool_fn(_feature_pool_type,kernel_size=(1,_conv1_pool),stride=(1,_conv1_pool)),
            activation_func(_activation),

            conv_bn(conv=conv2,in_channels=_conv1_out_channel,out_channels=_conv2_out_channel,bias=False),
            pool_fn(_feature_pool_type,kernel_size=(1,_conv2_pool),stride=(1,_conv2_pool)),
            activation_func(_activation),

            conv_bn(conv=conv3,in_channels=_conv2_out_channel,out_channels=_conv3_out_channel,bias=False),
            pool_fn(_feature_pool_type,kernel_size=(1,_conv3_pool),stride=(1,_conv3_pool)),
            activation_func(_activation),
        )

    def forward(self,x):
        x =  x.view(-1,self.in_size[1],1,self.in_size[2])
        x = self.blocks(x)
        #rint(x.shape)
        x = torch.flatten(x,start_dim=1)
        return x 

    @property
    def output_size(self):
        dummy =  torch.rand(size=self.in_size)
        out = self.forward(dummy.unsqueeze(0).to(next(self.parameters()).device))
        return out.shape 

class EEGCNNv3Decoder(nn.Module):
    def __init__(self,in_size, classes,options={},*args,**kwargs):
        super().__init__()
        ## Unpack options 
        _linear1_out = options.pop('linear1_out',80)
        _activation = options.pop('activation','relu')  
        _drop_rate = options.pop('drop_rate',0.7)   
        _avg_pool_size = options.pop('avg_pool_size',1)   
        if len(options) >0:
            extra = ', '.join('"%s"' % k for k in list(options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra) 
        self.batchnorm = nn.BatchNorm1d(_linear1_out)
        self.linear1 = nn.Linear(in_features=in_size[1],out_features=_linear1_out)
        self.linear2 = nn.Linear(in_features=_linear1_out, out_features=classes)
        self.activation = activation_func(_activation)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=_drop_rate)
        self.avgpool = nn.AvgPool1d(kernel_size=_avg_pool_size)
    def forward(self,x):
        x = self.linear1(x)
        x = self.activation(x) 
        #x = torch.square(x)
        #x = self.avgpool(x)
        #x = torch.log(x)
        #x = self.batchnorm(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x 



class EEGCNNv3(nn.Module):

    def __init__(self, in_channels=1, classes=4,input_size=(1,22,1000),encoder_opt={},
        decoder_opt={},*args, **kwargs):
        super().__init__()
        self.input_size = input_size
        ## Unpack **kwargs 
        self.encoder = EEGCNNv3Encoder(in_channels=in_channels,
            options=encoder_opt,input_size=input_size)
        encoder_out_size = self.encoder.output_size
        
        self.decoder = EEGCNNv3Decoder(in_size=encoder_out_size,classes=classes,
            options=decoder_opt)
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 

