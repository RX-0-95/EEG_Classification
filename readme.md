## About the code 

All result in the report are from subject1 .... subject9, subject_all and CNNv3.ipynb. The test accuracy of PSCNN is in CNNv3.ipynb. And rest of the test accuracy are in those subjects files. 

## Before run the code 

Before run the code, you need copy all files from data folder shared in ccle to the into data folder. The ccle only allow file  size less then 100MB, so I didn't include the data. Note, don't delete any file in the data foler, there are some downsampled file alredy there. 

Next, run the data_process.ipynb, this will generate will bandpassed data to the band_pass_data folder in the data folder. The data_process,ipynb will use the downsample data in the data foler and the data shared in ccle. 

## What's in each folder 

### eeg_net: 

contain pytorch model for PSCNN, 1D3LCNN, TEMSPACNN. There are also Residual CNN and other networks not in the report because I don't have time for parameter tunning. 

All CNN models are in the eeg_cnn.py file, the solver and other useful function are in the eeg_net_base.py, and util.py



### tensorflow_eeg 

contain CNN+LSTM in the eeg_net folder. 

### data 

The original data, processed data, and data_util.py for processing and extract the data. 