from eeg_net.eeg_net_base import* 
from eeg_net.eeg_cnn import *
from eeg_net.eeg_resnet import * 
from eeg_net.eeg_rnn import * 
import itertools as it
from torchsummary import summary


encoder_parm = {
    'gate_conv_sizes':[7,15,31],
    'gate_conv_out_channel':[30,60],
    'prob_conv_size':[6],
    'prob_down_sample':[2,3],
    'feature_conv_size':[7,31],
    'feature_conv_out_channel':[32,64],
    'feature_pool_type':['max','avg'],
    'feature_pool_size':[2,6],
    'activation': ['none','elu'],
}

class EEGCNN_solver():
    def __init__(self,model,data_dir,label_dir,model_encoder_opt={},
        model_decoder_opt={},train_opt={},*args,**kwargs):
        super().__init__() 
        self.model = model
        self.train_opt = train_opt
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.model_encoder_opt = model_encoder_opt 
        self.model_decoder_opt= model_decoder_opt 
        self.over_fit_counter = 0 
        self.over_fit_patience = 15
        self.log = {
            'param':[],
            'val_acc':[],
            'train_acc':[],
        }
    
    def write_log(self,filename ='param.txt'):
            f = open(filename,'w')
            for param, val_acc, train_acc in zip(self.log['param'],self.log['val_acc'],self.log['train_acc']):
                f.write(str(param)+':'+str(val_acc)+':'+str(train_acc)+'\n')
            f.close()

    def solve_param(self):
        my_dict = self.model_encoder_opt
        allNames = self.model_encoder_opt
        combinations = it.product(*(my_dict[Name] for Name in allNames))
        combinations_list = list(combinations)
  
        val_max = 0.0 
        for params in reversed(combinations_list): 
            torch.cuda.empty_cache()
            e_opt = {
                'gate_conv_size':params[0],
                'gate_conv_out_channel':params[1],
                'prob_conv_size':params[2],
                'prob_down_sample':params[3],
                'feature_conv_size':params[4],
                'feature_conv_out_channel':params[5],
                'feature_pool_type':params[6],
                'feature_pool_size':params[7],
                'activation': params[8],
            }
            e_model = self.model(1,4,encoder_opt=e_opt).to('cuda')
            loss_fn = nn.CrossEntropyLoss()
            #summary(e_model.cuda(),(1,22,1000))
            print('Start Training with param: {}'.format(params))
            logs,_= train(e_model, self.train_opt, loss_fn,
                data_dir=self.data_dir,
                label_dir=self.label_dir,
                preload_gpu=True)
            cur_val = np.max(logs['val_acc'])
            cur_train = np.max(logs['train_acc'])
            self.log['param'].append(params)
            self.log['val_acc'].append(cur_val)
            self.log['train_acc'].append(cur_train)
            if cur_val> val_max:
                val_max = cur_val

            print('current max val_acc:{} with param:{}'.format(val_max,params))
            self.write_log()
        return self.log
            

  
  