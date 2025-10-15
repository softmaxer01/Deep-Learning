import torch
import torch.nn as nn


class Depth_Wise_Conv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1,bias=False)
        #     for _ in range(in_channels)
        # ])
        # faster way
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3, 
            padding=1,
            bias=False,
            groups=in_channels)
    
    def forward(self, x):
        # outputs = []
        # for i, conv in enumerate(self.convs):
        #     outputs.append(conv(x[:, i:i+1, :, :])) 
        # return torch.cat(outputs, dim=1)
        return self.conv(x)
    

class Point_Wise_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
    
    def forward(self, x):
        return self.conv(x)
    
