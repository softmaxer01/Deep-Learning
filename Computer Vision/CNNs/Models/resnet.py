import torch
import torch.nn as nn

class Conv_layer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(**kwargs)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.conv(x))

class Residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, **kwargs):
        super().__init__()
        self.conv1 = Conv_layer(in_channels=in_c, out_channels=out_c, stride=stride, **kwargs)
        self.conv2 = Conv_layer(in_channels=out_c, out_channels=out_c, stride=1, **kwargs)
        
        if in_c != out_c or stride != 1:
            self.res_con = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride)
        else:
            self.res_con = nn.Identity()
    
    def forward(self, x):
        residual = self.res_con(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class ResNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.init_layer = self.create_init_layer(self.arch["initial"])
        self.conv_layer = self.create_conv_layer(self.arch["conv"])
        self.fc_layer = self.create_fc_layer(self.arch["fc"])
    
    def forward(self, x):
        out1 = self.init_layer(x)
        out2 = self.conv_layer(out1)
        out3 = self.fc_layer(out2)
        return out3
    
    def create_conv_layer(self, arch):
        layer = []
        for i in arch:
            if type(i) == tuple:
                in_c, out_c, kernel_size, num_blocks, stride, padding = i
                layer.append(
                    Residual_block(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
                )
                for j in range(1, num_blocks):
                    layer.append(
                        Residual_block(out_c, out_c, kernel_size=kernel_size, stride=1, padding=padding)
                    )
        return nn.Sequential(*layer)
    
    def create_init_layer(self, arch):
        layer = []
        for i in arch:
            if type(i) == tuple:
                layer.append(nn.Conv2d(i[0], i[1], i[2], i[3], padding=i[4] if len(i) > 4 else 0))
            if type(i) == str and i == "M":
                layer.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layer)
    
    def create_fc_layer(self, arch):
        layer = []
        for i in arch:
            if type(i) == str and i == "A":
                layer.append(nn.AdaptiveAvgPool2d(1))
            if type(i) == str and i == "F":
                layer.append(nn.Flatten(start_dim=1))
            if type(i) == tuple:
                layer.append(nn.Linear(in_features=i[0], out_features=i[1]))
            if type(i) == str and i == "R":
                layer.append(nn.ReLU())
        return nn.Sequential(*layer)


