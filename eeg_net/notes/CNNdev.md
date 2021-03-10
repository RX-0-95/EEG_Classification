# CNN Design 

## CNN V2 



## CNN V3 (also used as part of the CNN+LSTM model)
1. conv1x16: pad same, stride 1, in_channel: 22, out_channel: 32
2. relu 
3. maxpool 5, stride 5 

1. conv1x16: pad same, stride 1, in_channel: 64, out: 128
2. relu 
3. maxpool 5, stride 5 

1. conv1x16: pad same, stride 1 : in:128, out: 256
2. relu 
3. maxpool 5, stride 5


1. fc net (8*256 to 4) 


### Param 

0.71 
encoder_opt = {
    'conv1_size': 3,
    'conv1_out_channel': 22,
    'conv2_size': 3,
    'conv2_out_channel': 44,
    'conv3_size': 3,
    'conv3_out_channel': 22,
    'conv1_pool':2,
    'conv2_pool':2,
    'conv3_pool':2,
    'activation': 'none',
    'pool_type': 'max'
    
}
decoder_opt={
    'drop_rate': 0.7, 
    'linear1_out': 50, 
    'activation': 'leaky_relu',
}
* band pass data 0.1-45 Hz (best)

                    encoder_opt = {
                    'conv1_size': 5,
                    'conv1_out_channel': 22,
                    'conv2_size': 5,
                    'conv2_out_channel': 44,
                    'conv3_size': 5,
                    'conv3_out_channel': 22,
                    'conv1_pool':2,
                    'conv2_pool':2,
                    'conv3_pool':2,
                    'activation': 'none',
                    'pool_type': 'max'
                    
                }
                decoder_opt={
                    'drop_rate': 0.8, 
                    'linear1_out': 64, 
                    'activation': 'leaky_relu',
                    'avg_pool_size':4
                }


                train_options = {
                    'train_batch_size': 64,
                    'scheduler_patience': 150,
                    'scheduler_factor': 0.5,
                    'weight_decay': 0.15,
                    'val_batch_size': 1,
                    'learning_rate': 1e-4,
                    'epoch_num': 400,
                    'downsample_split': True,
                }

* Normalize the data help reduce overfitting 