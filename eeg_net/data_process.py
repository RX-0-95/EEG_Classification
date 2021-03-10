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

class transfer_data(object):
    def __init__(self,data, file_name) :
        super().__init__()
        self.data = data 
        self.file_name = file_name
    def save_data(self,data):
        with open(self.file_name,'wb') as f:
            np.save(f,data)

    def transfer(self):
        return self.data 
    
    def transfer_and_save(self):
        self.save_data(self.transfer())

class norm_transfer(transfer_data):
    def __init__(self, data, file_name,norm_factor=35):
        super().__init__(data, file_name)
        self.norm_factor = norm_factor
    
    def transfer(self):
        b,w,h = self.data.shape
        data = self.data/self.norm_factor
        return data  

class fft_tansfer(object):
    def __init__(self, data, file_name):
        super().__init__()
        self.data = data 
        self.file_name = file_name
    
    def transfer(self):
        b,w,h = self.data.shape
        phase_data = np.zeros_like(self.data)
        mag_data = np.zeros_like(self.data)

        for i in np.arange(self.data.shape[0]):
            cur_data = self.data[i]
            for j,sig in enumerate(cur_data):
                rt_data=np.fft.fft(sig)
                phase_data[i,j] = np.abs(rt_data)
                mag_data[i,j] = np.angle(rt_data)
      
        return phase_data, mag_data
    
    def save_data(self,data):
        phase_data,mag_data = data 
        phase_dir = self.file_name+'phase'
        mag_dir = self.file_name+'mag'
        with open(phase_dir,'wb') as f:
            np.save(f,phase_data)
        with open(mag_dir,'wb') as f:
            np.save(f,mag_data)
    
    def transfer_and_save(self):
        data = self.transfer() 
        self.save_data(data)