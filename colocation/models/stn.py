import torch
import torch.nn as nn
# from pynvml import *

class STN(nn.Module):
    def __init__(self, dropout_rate, input_channels):
        super(STN, self).__init__()
        self.d = dropout_rate
        # sensor* 32*~7000
        self.dropout1 = nn.Dropout(self.d)
        self.conv1 = nn.Sequential( 
            nn.Conv1d(in_channels = input_channels, out_channels = 128, kernel_size = 8, stride = 3, padding = 0),
            nn.ELU(),  
            nn.MaxPool1d(kernel_size = 3, stride = 2)
        )
        self.dropout4 = nn.Dropout(self.d)
        self.conv2 = nn.Sequential( 
            nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 7, stride = 3, padding = 0),
            nn.ELU(),  
            nn.MaxPool1d(kernel_size = 3, stride = 2)
        )
        self.dropout2 = nn.Dropout(self.d)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels = 256, out_channels = 128, kernel_size = 6, stride = 1, padding = 0),
            nn.ELU(),  
            nn.MaxPool1d(kernel_size = 3, stride = 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels = 128, out_channels = 1, kernel_size = 1, stride = 1),
        )
        self.dropout3 = nn.Dropout(self.d)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.dropout4(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout3(x)
        norm = x.norm(dim=1, p=2, keepdim = True)
        x = x.div(norm.expand_as(x))
        return x
