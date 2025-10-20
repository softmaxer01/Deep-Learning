import torch
import torch.nn as nn

class conv_layer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(**kwargs)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv(x))

class inception_layer(nn.Module):
    def __init__(self, layer_arch):
        super().__init__()
        self.layer_arch = layer_arch
        self.branches = self.create_layer(self.layer_arch)
    
    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)  
    
    def create_layer(self, layer_arch):
        layer = [[], [], [], []]
        branch_idx = 0
        

        for j in layer_arch:
            if type(j) == tuple:
                layer[branch_idx].append(conv_layer(in_channels=j[0],
                                                    out_channels=j[1],
                                                    kernel_size=j[2],
                                                    stride=j[3],
                                                    padding=j[4]))
                branch_idx += 1
                
            if type(j) == list:
                layer[branch_idx].append(
                    nn.Sequential(
                        conv_layer(in_channels=j[0][0],
                                    out_channels=j[0][1],
                                    kernel_size=j[0][2],
                                    stride=j[0][3],
                                    padding=j[0][4]),
                        conv_layer(in_channels=j[1][0],
                                    out_channels=j[1][1],
                                    kernel_size=j[1][2],
                                    stride=j[1][3],
                                    padding=j[1][4])
                    )
                )
                branch_idx += 1
                
            if type(j) == str:
                layer[branch_idx].append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        
        return nn.ModuleList([nn.Sequential(*b) for b in layer if b])
    

class GoogleNet(nn.Module):
    def __init__(self,arch,linear_layers):
        super().__init__()
        self.arch = arch
        self.conv_layer = self.create_net(self.arch)
        self.linear_layer = self.create_linear(linear_layers)
    
    def forward(self,x):
        return self.linear_layer(self.conv_layer(x))
    
    def create_net(self,arch):
        network = []
        for i in arch:
            if type(i) == list:
                network.append(inception_layer(i))
            if type(i) == str:
                network.append(
                    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
                )
            if type(i) == tuple:
                network.append(conv_layer(
                        in_channels = i[0],
                        out_channels = i[1],
                        kernel_size = i[2],
                        stride = i[3],
                        padding = i[4]
                    )
                )
        return nn.Sequential(*network)
    
    def create_linear(self, linear_layers):
        layer = []
        for i in linear_layers:
            if type(i) == str:
                if i == "A":
                    layer.append(nn.AdaptiveAvgPool2d((1, 1)))  
                if i == "D":
                    layer.append(nn.Dropout(0.5))
            if type(i) == tuple:
                layer.append(nn.Flatten())  
                layer.append(nn.Linear(in_features=i[0], out_features=i[1]))
        return nn.Sequential(*layer)




# img = torch.randn((1,3,224,224))
# model = GoogleNet(inception_net,linear_layers)
# print(model(img).shape)
