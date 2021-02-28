import numpy 
import torch
from torch._C import device 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchvision import transforms, utils 
import time 

device = torch.device('cuda0' if torch.cuda.is_available() else 'cpu')
print(device)

'''
ShallowConv: 
ShallowConv is given by TA as the baseline of this project 
'''

class ShallowConv(nn.Module):
    def __init__(self,in_channels, classes):
        super(ShallowConv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 40,(1,25),stride=1)
        self.fc1 = nn.Linear(880,40)
        