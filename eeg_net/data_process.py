import numpy as np 
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def split_data(data,downsampel_rate=2):
    data_len = len(data)//downsampel_rate
    d1 = np.zeros((data_len,))
    d2 = np.zeros((data_len,))
    for i in np.arange(len(data)):
        if i%2==0:
            d1[int(i/downsampel_rate)]=data[i]
        else:
            d2[int(i//downsampel_rate)] =data[i]
    return d1,d2 

def downsample(data,lable,data_filename='X.npy',label_filename='y.npy',downsample_rate=2):
    b,w,h = data.shape 
    rt_data = np.zeros((downsample_rate*b,w,h//downsample_rate))
    rt_label = np.zeros(downsample_rate*len(lable))
    for i in np.arange(data.shape[0]):
        cur_data = data[i]
        #print(cur_data.shape)
        for j,sig in enumerate(cur_data):
            d1,d2= split_data(sig)
            rt_data[downsample_rate*i,j]=d1 
            rt_data[downsample_rate*i+1,j]=d2      
        rt_label[downsample_rate*i] = lable[i]
        rt_label[downsample_rate*i+1]=lable[i]
    with open(data_filename,'wb') as f:
        np.save(f,rt_data)
    with open(label_filename,'wb') as f:
        np.save(f,rt_label)

    return rt_data, rt_label

def band_pass_data(data,low_cut,high_cut,fs,data_filename='band_X.npy',):
    b,w,h = data.shape
    rt_data = np.zeros_like(data)
    for i in np.arange(data.shape[0]):
        cur_data = data[i]
        for j,sig in enumerate(cur_data):
            rt_data[i,j]=butter_bandpass_filter(
                sig,lowcut=low_cut,highcut=high_cut,fs=fs)
    
    with open(data_filename,'wb') as f:
        np.save(f,rt_data)