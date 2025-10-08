import torch 
import torch.nn as nn
import torch.optim as optim 

class LeNet(nn.Module):
    def __init__(self, img_dim=28, num_channels=1):
        super().__init__()
        self.img_dim = img_dim
        self.num_channels = num_channels

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # Fully connected layers 
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5*5*16, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    def forward(self, img):
        x = self.conv_layers(img)
        x = self.fc_layers(x)
        return x
    


# image = torch.randn((4,3,28,28))
# lenet = LeNet()
# print(lenet(image).shape)