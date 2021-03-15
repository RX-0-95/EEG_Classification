from os import scandir
import os 
from typing import Sequence
import numpy as np
from tensorflow.python.ops.gen_array_ops import parallel_concat
import torch

import torch.nn as nn
from torch.types import Device
from torch.utils import data 
from torch.utils.data import Dataset, DataLoader,TensorDataset, random_split
from torch.utils.data.dataset import Subset 
from torchvision import transforms, utils 
import torch.optim as optim 
import time 
from itertools import accumulate
from livelossplot import PlotLosses 
import matplotlib.pyplot as plt 
from eeg_net.utils import *
from eeg_net.data_process import * 
'''
Process Data 
TODO: Fourier transfer magnitude and phase  
'''


class Conv2dAuto(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)

class Conv1dAuto(nn.Conv1d):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0]//2)

class Conv2dNopad(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

def activation_func(activation):
    return nn.ModuleDict([
        ['relu',nn.ReLU(inplace=False)],
        ['leaky_relu',nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu',nn.SELU(inplace= True)],
        ['elu',nn.ELU(inplace=False)],
        ['none', nn.Identity()]
    ])[activation]



class EEGDataset(Dataset):
    """EEG dataset."""
    def __init__(self, subset, transform=None):        
        self.transform = transform
        self.subset = subset 
    
    def __getitem__(self, index):
 
        x, y = self.subset[index]
        if self.transform:
            if self.transform == 'shift_positive':
               x = self.shift_positive(x) 
            elif self.transform =='square':
                x = self.square(x)
            elif self.transform =='fft':
                print(x.shape)
                for i,data in enumerate(x):
                    print(data.shape)
                    x[i] = butter_bandpass_filter(data,3,25,250) 
            elif self.transform =='none':
                pass 
        return x, y
        
    def __len__(self):
        return len(self.subset)

    def shift_positive(self,x):
        min_value = torch.min(x)
        rt = x - min_value 
        return rt 
    def square(self,x): 
        rt = torch.square(x)
        return rt 
    
    def fft(self,x):
        return x 
    

class ShallowConv(nn.Module):
    '''
    ShallowConv: 
    ShallowConv is given by TA as the baseline of this project 
    '''
    def __init__(self,in_channels, classes,device=None,*args,**kwargs):
        super().__init__()
        # Unpack the kwargs 
        _activation = kwargs.pop('activation','none')
        self.device = self.set_device(device) 
        self.activation = activation_func(_activation)
        self.conv1 = nn.Conv2d(in_channels, 40,(1,25),stride=1)
        self.fc1 = nn.Linear(880,40)
        self.avgpool = nn.AvgPool1d(75, stride=15)
        self.fc2 = nn.Linear(2440,classes)
        self.softmax= nn.Softmax(dim=1) 
        self.drop = nn.Dropout(0.6)
    def forward(self,x):
        x = x.view(-1,1,22,1000)
        x = self.conv1(x)
        #x = self.activation(x)
        x = x.permute(0,3,1,2)
        x = x.view(-1,976,880)
        x = self.fc1(x)
        x = self.activation(x)
        x = torch.square(x)
        x = x.permute(0,2,1)
        x = self.avgpool(x)
        x = torch.log(x)
        x = x.reshape(-1,40*61)
        x =self.drop(x)
        x = self.fc2(x)
        x = self.softmax(x) 
        return x 

    def set_device(self,device=None):
        if device == None:
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            _device = torch.device(device)
        self.device = _device 
        return _device 


class ShallowConv2(nn.Module):
    '''
    ShallowConv: 
    ShallowConv is given by TA as the baseline of this project 
    '''
    def __init__(self,in_channels, classes,device=None,*args,**kwargs):
        super().__init__()
        # Unpack the kwargs 
        _activation = kwargs.pop('activation','none')
        self.device = self.set_device(device) 
        self.activation = activation_func(_activation)
        self.conv1 = nn.Conv2d(in_channels, 40,(1,25),stride=1)
        self.fc1 = nn.Linear(880,40)
        self.avgpool = nn.AvgPool1d(75, stride=15)
        self.fc2 = nn.Linear(40*27,classes)
        self.softmax= nn.Softmax(dim=1) 
        self.drop = nn.Dropout(0.2)
    def forward(self,x):
        x = x.view(-1,1,22,500)
        x = self.conv1(x)
        #x = self.activation(x)
        x = x.permute(0,3,1,2)
        #print(x.shape)
        x = x.view(-1,476,880)
        x = self.fc1(x)
        x = self.activation(x)
        x = torch.square(x)
        x = x.permute(0,2,1)
        x = self.avgpool(x)
        x = torch.log(x)
        #print(x.shape)
        x = x.reshape(-1,40*27)
        #x = self.drop(x)
        x = self.fc2(x)
        x = self.softmax(x) 
        return x 

    def set_device(self,device=None):
        if device == None:
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            _device = torch.device(device)
        self.device = _device 
        return _device 
"""
class ShallowConvEncoder(nn.Module):
    def __init__(self,in_channels,*args,**kwargs):
        super().__init__()
        _activation = kwargs.pop('activation','none')
"""     

class DsShallowConv(nn.Module):
    '''
    
    '''
    def __init__(self,in_channels, classes,device=None,*args,**kwargs):
        super().__init__()
        # Unpack the kwargs 
        _activation = kwargs.pop('activation','none')
        self.device = self.set_device(device) 
        self.activation = activation_func(_activation)
        #self.conv1 = nn.Conv2d(in_channels, 40,(1,13),stride=1)
        #self.conv1 = Conv2dAuto(in_channels,40,(1,13),stride=1)
        self.conv1 = Conv2dAuto(in_channels=in_channels,
                                out_channels = 40,
                                kernel_size=(1,13),
                                stride = 1)
        
        self.fc1 = nn.Linear(880,40)
        self.avgpool = nn.AvgPool1d(75, stride=15)
        self.fc2 = nn.Linear(40*29,classes)
        self.softmax= nn.Softmax(dim=1) 
        self.drop = nn.Dropout(0.15)
    def forward(self,x):
        x = x.view(-1,1,22,500)
        x = self.conv1(x)
        #x = self.activation(x)
        x = x.permute(0,3,1,2)
        #print(x.shape)
        x = x.view(-1,500,880)
        x = self.fc1(x)
        x = self.activation(x)
        x = torch.square(x)
        x = x.permute(0,2,1)
        x = self.avgpool(x)
        x = torch.log(x)
        #print(x.shape)
        x = x.reshape(-1,40*29)
        #x = self.drop(x)
        x = self.fc2(x)
        x = self.softmax(x) 
        return x 

    def set_device(self,device=None):
        if device == None:
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            _device = torch.device(device)
        self.device = _device 
        return _device 


def eeg_train_val_loader(_data_dir, _label_dir,*args, **kwargs):
    """ 
    Load the data, splilt the train set and validation set 
    All Data will be load to "device". Suit for project have small 
    data size. Larger data size may cause the device run out of ram.
    In that case, consider load the data from disc 
    Required argument:
    - _data_dir: directory of the data 
    - _label_dir: directory of the label 

    Optional argument: 
    - split_ratio: the ratio of train and validation data, default: 0.8 
    - train_batch_size: batch size of train data, default: 32
    - val_batch_size: batch size of validation data, default: 8
    - device: All data will load to this device, default: None (ie.cuda when possible)

    Return: 
    dict "train", "val" 
    contain pytorch DataLoder object
    """
    data_dir = _data_dir 
    label_dir = _label_dir  

    #Unpack keyword argument 
    _kwargs = kwargs.copy() 
    split_ratio= _kwargs.pop('split_ratio',0.8)
    train_batch_size = _kwargs.pop('train_batch_size',32)
    val_batch_size = _kwargs.pop('val_batch_size',8)
    device = get_device(_kwargs.pop('device',None)) 
    transform = _kwargs.pop('transform',None)
    downsample_split = _kwargs.pop('downsample_split',False)
    # Throw an error if extra keyword 
    if len(_kwargs) >0:
        extra = ', '.join('"%s"' % k for k in list(_kwargs.keys()))
        raise ValueError('Unrecognized arguments %s' % extra)
    ## Check if the pass in parameter is director to the data, or data it self
    if isinstance(data_dir,str):
        eeg_data = np.load(data_dir)
        eeg_label = np.load(label_dir)
    else:
        eeg_data = data_dir
        eeg_label = label_dir
    
    #Reform the label to 0,1,2,3
    eeg_label -= 769 
    assert eeg_label.shape[0]==eeg_data.shape[0], \
        "The first size of label and data must be same."
    eeg_label = torch.from_numpy(eeg_label).float().long().to(device)
    eeg_data = torch.from_numpy(eeg_data).float().to(device)

    init_dataset = TensorDataset(eeg_data,eeg_label)
    train_length =  int(len(init_dataset)*split_ratio)
    lengths = [train_length,int(len(init_dataset)-train_length)] 
    if downsample_split == False: 
        subset_train, subset_val = random_split(init_dataset, lengths) 
    else:
        subset_train, subset_val = downsample_random_split(init_dataset, lengths) 
        
    train_data = EEGDataset(
        subset_train, transform=transform)

    val_data = EEGDataset(
        subset_val, transform=transform)
    #print(len(val_data))
    #print(len(train_data))
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_data, batch_size=train_batch_size, shuffle=True, num_workers=0),
        'val': torch.utils.data.DataLoader(
            val_data, batch_size=val_batch_size, shuffle=False, num_workers=0)
    }
    return dataloaders 

def eeg_test_loader(_data_dir,_label_dir,downsample_r=None,*args,**kwargs):
    #Unpack keyword argument 
    _kwargs = kwargs.copy() 
    device = get_device(_kwargs.pop('device',None)) 
    transform = _kwargs.pop('transform',None)
    
    # Throw an error if extra keyword 
    if len(_kwargs) >0:
        extra = ', '.join('"%s"' % k for k in list(_kwargs.keys()))
        raise ValueError('Unrecognized arguments %s' % extra)
  
    if isinstance(_data_dir,str):
        eeg_data = np.load(_data_dir)
        eeg_label = np.load(_label_dir)
    else:
        eeg_data = _data_dir
        eeg_label = _label_dir
    eeg_label -= 769 
    eeg_label = torch.from_numpy(eeg_label).float().long().to(device)
    eeg_data = torch.from_numpy(eeg_data).float().to(device)
    eeg_dataset = TensorDataset(eeg_data,eeg_label)
    test_data = EEGDataset(
        eeg_dataset,transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1,shuffle = False, num_workers = 0 
    )
    return test_loader 

def downsample_random_split(dataset, lengths):
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    rng = np.random.default_rng() 
    indices1 = np.arange(int(sum(lengths)/2))
    #indices1 = rng.integers(low=0,high=(sum(lengths)/2))
    np.random.shuffle(indices1)
    indices1 = indices1*2
    indices2 = indices1+1
    c_indices = npcross_combine(indices1,indices2).tolist() 
    return [Subset(dataset, c_indices[offset - length : offset]) for offset, length in zip(accumulate(lengths), lengths)]

def npcross_combine(arr1,arr2):
    len = arr1.shape[0] + arr2.shape[0]
    rtarr = np.zeros((len,)) 
    for i in np.arange(arr1.shape[0]):
        
        rtarr[2*i]=arr1[i]
        rtarr[2*i+1] =arr2[i]
    return rtarr.astype(int)


class OverfitDetector():
    def __init__(self,threshold=0.65,patience=20,verbose =True):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.counter = 0 
        self.verbose = verbose
        self.overfit = False 
    def step(self,train_acc,val_acc):
        if train_acc*self.threshold>val_acc:
            self.counter+=1 
        if self.counter>self.patience:
            self.counter = 0 
            self.overfit = True
            if self.verbose:
                print('Overfit is detected!!!')
        else:
            self.overfit = False 

def train(model,options,criterion,
    data_dir,label_dir,
    device=None,preload_gpu=False,
    verbose=True,plot_graph=True,l2_reg =False):
    # log record loss 
    logs = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [] 
    } 

    model.train() 
    train_device = get_device(device)
    _options = options.copy() 
    # Unpack options 
    _train_val_split_ratio = _options.pop('train_val_split_ratio',0.8)
    _train_batch_size = _options.pop('train_batch_size',32)
    _val_batch_size = _options.pop('val_batch_size',8)
    _learning_rate = _options.pop('learning_rate',1e-4)
    _weight_decay = _options.pop('weight_decay',0.001)
    _scheduler_factor = _options.pop('scheduler_factor',0.8)
    _scheduler_patience = _options.pop('scheduler_patience',50)
    _transform = _options.pop('transform',None)
    _overfit_threshold = _options.pop('overfit_threshold',0.65)
    _overfit_patience = _options.pop('overfit_patience',20)
    _downsample_split = _options.pop('downsample_split',False)
    _save_flag = _options.pop('save_flag',False)
    _min_save_epoch = _options.pop('min_save_epoch',50)
    _model_dir = _options.pop('model_dir','model')
    _model_name= _options.pop('model_name','model.pth')
    epoch_num = _options.pop('epoch_num',250)

    # Check if there is unexpected options 
    if len(_options) >0:
        extra = ', '.join('"%s"' % k for k in list(_options.keys()))
        raise ValueError('Unrecognized arguments in options%s' % extra)

    data_loaders = eeg_train_val_loader(
        data_dir,label_dir,
        device = train_device if preload_gpu else 'cpu',
        split_ratio = _train_val_split_ratio,
        train_batch_size =_train_batch_size,
        val_batch_size = _val_batch_size,
        transform = _transform,
        downsample_split = _downsample_split) 

    train_loader = data_loaders['train']
    val_loader = data_loaders['val']

    optimizer = optim.Adam(
        model.parameters(), 
        lr = _learning_rate,
        weight_decay= _weight_decay
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode= 'min',
        factor=_scheduler_factor,
        patience= _scheduler_patience,
        verbose=True,
        threshold=1e-2 
        )
    overfit_detector = OverfitDetector(_overfit_threshold,_overfit_patience)
    print('Start training...')
    print('Epoch\tTrain Loss\tTrain Acc\tTest Loss\tTest_Acc\t')
    best_metric = 0.0 
    for i in np.arange(epoch_num):
        epoch_loss = [] 
        epoch_metric = [] 
        
        iter_ct = 0 
        
        for data in train_loader:
            #copy all tensor to GPU, donothing if the alread in GPU 
            x, y = data 
           
            if not preload_gpu:
                x = x.to(train_device)
                y = y.to(train_device,dtype =torch.long)
            #assert x.is_cuda == False, "X is not cuda"

            #clear the existing gradients 
            optimizer.zero_grad() 

            #Forward pass 
            y_hat = model(x)
            loss = criterion(y_hat,y)
            loss.backward() 
            # Update the weights 
            optimizer.step() 
            epoch_loss.append(loss.data.item())
            yhat_detacted = y_hat.detach()
            #print(type(yhat_detacted))
            y_detacted = y.detach()  
            #y_pred = torch.argmax(yhat_detacted,dim=1)
            epoch_metric.append(get_pred_acc(yhat_detacted,y_detacted))
            iter_ct +=1 
            #print(iter_ct)
            if verbose: 
                if iter_ct%50 ==50-1:
                    print('--Iter %d\t%4.6f' %(
                        iter_ct,
                        loss))

        avg_epoch_loss = sum(epoch_loss)/len(epoch_loss)
        avg_epoch_metric = sum(epoch_metric)/len(epoch_metric)
        avg_val_loss, avg_val_metric = test_net(model,val_loader,
                                                criterion,train_device)
              
        ## log the result 
        logs['train_loss'].append(avg_epoch_loss) 
        logs['train_acc'].append(avg_epoch_metric)
        logs['val_loss'].append(avg_val_loss)
        logs['val_acc'].append(avg_val_metric)
        #update best metric and save the model 
        if avg_val_metric >= best_metric:
            best_metric = avg_val_metric 
            if _save_flag and i>_min_save_epoch:
                torch.save(
                    model.state_dict(),
                    os.path.join(_model_dir,_model_name)
                )
                print('Saving model...')
        ## Print result 
        if verbose:
            print('%d\t%4.6f\t%4.6f\t%4.6f\t%4.6f\t' % (
                i, 
                avg_epoch_loss, 
                avg_epoch_metric, 
                avg_val_loss, 
                avg_val_metric
                ))
        scheduler.step(avg_val_metric) 
        overfit_detector.step(avg_epoch_metric,avg_val_metric) 
        if overfit_detector.overfit:
            print('Detect Model overfit, stop training....')
            break; 

    if plot_graph:
        plot_logs(logs)        

    return logs, model 

def get_pred_acc(output,gt):
    _,pred = output.max(1)
    correct_pred = pred.eq(gt).sum().item() 
    return correct_pred/len(gt)

def test_net(model,test_loader,criterion,device):
    model.eval() 

    test_loss = [] 
    test_metrics = [] 

    with torch.no_grad():
        for data in test_loader:
            x,y = data
            x = x.to(device)
            y = y.to(device)
            
            #Forward pass 
            y_hat = model(x)
            #print(y)
            #print(y_hat)
            loss = criterion(y_hat,y)
            #print(loss)
            #print(loss)
            #for param in model.parameters():
            #    print(param.data)
            #    break
            #loss_detached = loss.detach() 
            #print(loss.shape)
            test_loss.append(loss.data.item())
            yhat_detacted = y_hat.detach()
            y_detacted = y.detach()  
            test_metrics.append(get_pred_acc(yhat_detacted,y_detacted))
    model.train() #set back to train mode 
    # Print result 
    avg_loss = sum(test_loss) / len(test_loss)
    avg_metric = sum(test_metrics) / len(test_metrics)
    return avg_loss, avg_metric

def net_pred(model,test_loader,criterion,device):
    model.eval() 
    pred_list = []
    with torch.no_grad():
        for data in test_loader:
            x,y = data
            x = x.to(device)
            y = y.to(device)
            
            #Forward pass 
            y_hat = model(x)
            #print(y)
            #print(y_hat)
            loss = criterion(y_hat,y)
            
            yhat_detacted = y_hat.detach()
            _,pred = yhat_detacted.max(1)
            #y_hat_data = y_hat.data.cpu().numpy()
            pred_list.append(pred.data.item())
    model.train() #set back to train mode 
    
    return pred_list

def plot_logs(logs):
    fig,(ax1,ax2) = plt.subplots(1,2) 
    ax1.plot(logs['train_loss'],label='Train')
    ax1.plot(logs['val_loss'],label = 'Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.plot(logs['train_acc'],label = 'Train')
    ax2.plot(logs['val_acc'],label = 'Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    fig.legend() 
    plt.show() 

def avg_test_acc(model,input_size, data_dir,loss_fn,train_opt={},model_opt={},trails = 10):

    test_acc = []
    for i in np.arange(trails):
        X_train = data_dir['X_train_dir'].copy()
        y_train = data_dir['y_train_dir'].copy()
        X_test = data_dir['X_test_dir'].copy()
        y_test = data_dir['y_test_dir'].copy()
        model_net  = model(input_size = input_size,options = model_opt).to('cuda')
        #model_net  = model().to('cuda')
        train(model_net,train_opt,loss_fn,
                data_dir=X_train,
                label_dir=y_train,
                preload_gpu= True)
        test_loader = eeg_test_loader(X_test,y_test)
        avg_loss, acc = test_net(model_net,test_loader,loss_fn,'cuda')
        print('Test accuracy in trail {}: {}'.format(i,acc))
        test_acc.append(acc)
        torch.cuda.empty_cache()
        del model_net
    return test_acc, sum(test_acc)/len(test_acc)    

    