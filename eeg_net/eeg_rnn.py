from eeg_net.eeg_net_base import * 









class EEGRNN(nn.Module):
    def __init__(self,in_channels=1, classes=4, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d((1,40),padding='none')


    def forward(self,x):
        x = self.conv1(x)