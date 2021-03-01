import torch

def get_device(device=None):
    if device == None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _device = torch.device(device)
    return _device 
