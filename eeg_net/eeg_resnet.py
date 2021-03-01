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

class ResConv(nn.Module):
    pass 



