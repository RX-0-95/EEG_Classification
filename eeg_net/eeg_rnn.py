from eeg_net.eeg_cnn import conv_bn
from eeg_net.eeg_resnet import *
from eeg_net.eeg_net_base import *
from eeg_net.eeg_cnn import * 
from functools import partial
from torch.nn.modules import dropout
from eeg_net.eeg_net_base import * 
from torch import nn 


class CNNEEGLSTM(nn.Module):
    def __init__(self,in_channels, num_classes=4,options={}):
        super().__init__() 
        _conv1_size = options.pop('conv1_size',3)
        _conv1_out_channel = options.pop('conv1_out_channel',128)
        _conv1_pool = options.pop('conv1_pool',4)
        _conv2_size = options.pop('conv2_size',3)
        _conv2_out_channel = options.pop('conv2_out_channel',64)
        _conv2_pool = options.pop('conv2_pool',2)
        _feature_pool_type = options.pop('pool_type','max')
        _activation = options.pop('activation','none')
        _lstm_hidden_size = options.pop('lstm_hidden_size',256)
        _lstm_drop_rate = options.pop('lstm_drop_rate',0.0)
        _lstm_layer = options.pop('lstm_layer',2)

        conv1 = partial(Conv1dAuto,kernel_size=_conv1_size,stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=_conv1_pool,stride=_conv1_pool)
        conv2 = partial(Conv2dAuto,kernel_size=(1,_conv2_size),stride = 1)
        self.lstm = nn.LSTM(input_size=_conv1_out_channel,
                            hidden_size = _lstm_hidden_size,
                            dropout=_lstm_drop_rate,
                            num_layers =_lstm_layer,
                            batch_first = True)
        self.fc = nn.Linear(in_features=_lstm_hidden_size,
                                out_features=num_classes)
        if len(options) >0:
            extra = ', '.join('"%s"' % k for k in list(options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra)
            
        #self.conv = conv_bn(conv=conv1,in_channels=in_channels,out_channels=_conv1_out_channel,bias=False)
        self.conv1 = nn.Conv1d(in_channels =in_channels, 
                    out_channels = _conv1_out_channel,
                    kernel_size=_conv1_size,stride=1)
        self.conv1_norm = nn.BatchNorm1d(_conv1_out_channel)
        self.activation = activation_func(_activation)
        self.softmax = nn.Softmax(dim=1)
        self.conv_layer = nn.Sequential(
            conv_bn(conv=conv1,in_channels=in_channels,out_channels=_conv1_out_channel,bias=False),
            #pool_fn(_feature_pool_type,kernel_size=(1,_conv1_pool),stride=(1,_conv1_pool)),
            self.pool1,
            activation_func(_activation),
        )
        
        self.lstm_layer = nn.Sequential(
            self.lstm 
        )
        self.fc_layer = nn.Sequential(
            self.fc 
        )
    def forward(self,x):
        #x = self.conv_layer(x)
        x = self.conv1(x)
        x = self.conv1_norm(x)
        x = self.activation(x)
        x =self.pool1(x)
        #print(x.shape)
        x = x.permute(0,2,1)
        x,_ = self.lstm_layer(x)
        x = self.fc_layer(x[:,-1,:])
        #print(x.shape)
        x = self.softmax(x)
        #print(x)
        return x 
    


class NativeEEGLSTM(nn.Module):
    '''
    Targe time step:
    '''
    def __init__(self,input_size, hidden_size,num_layers,num_classes,device='cuda', *args, **kwargs):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.0)
        self.fc1 = nn.Linear(hidden_size,num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = x.permute(0,2,1)
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        state = (h0,c0)
        out, state = self.lstm(x,state) 
        out = self.fc1(out[:,-1,:])
        #print(out[:,-1,:].shape)
        #print(out.shape)
        out = self.softmax(out)
        return out 

class EEGLSTM(nn.Module):
    def __init__(self,input_size, hidden_size,num_layers,num_classes,device='cuda', *args, **kwargs):
        """
        Input_size: the 
        """
        super().__init__()
        self.device = device
        #self.conv1 = nn.Conv2d((1,40),padding='none')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,num_classes)
        self.softmax = nn.Softmax(dim=1)
        #self.fc2 = nn.Linear()
    def forward(self,x):
        #x = x.permute(0,2,1)
        x = x.view(-1,100,22*10)
        #x = x.permute(0,2,1)
        #print(x.size(0))
        #print(x.shape)

        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        state = (h0,c0)
        out, state = self.lstm(x,state) 
        out = self.fc1(out[:,-1,:])
        #print(out[:,-1,:].shape)
        #print(out.shape)
        #out = self.softmax(out)
        return out 