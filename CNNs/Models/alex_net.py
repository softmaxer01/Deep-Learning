import torch
import torch.nn as nn

# Convolutional architecture: (in_channels, out_channels, kernel_size, stride, padding)
architecture_config = [
    (3, 96, 11, 4, 0),
    "M",
    (96, 256, 5, 1, 2),
    "M",
    (256, 384, 3, 1, 1),
    (384, 384, 3, 1, 1),
    (384, 256, 3, 1, 1),
    "M",
]

# Fully connected architecture: (in_features, out_features) and "R" for ReLU
fc_layer_config = [(256 * 5 * 5, 4096), "R", (4096, 4096), "R", (4096, 10)]


class Conv_block(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(**kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_architecture = architecture_config
        self.fc_arch = fc_layer_config
        self.conv_layer = self.create_conv_layer(self.conv_architecture)
        self.fc = self.create_fc_layer(self.fc_arch)

    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def create_conv_layer(self, architecture):
        layers = []
        for x in architecture:
            if isinstance(x, tuple):
                layers.append(
                    Conv_block(
                        in_channels=x[0],
                        out_channels=x[1],
                        kernel_size=x[2],
                        stride=x[3],
                        padding=x[4],
                    )
                )
            elif isinstance(x, str) and x == "M":
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        return nn.Sequential(*layers)

    def create_fc_layer(self, architecture):
        layers = []
        for x in architecture:
            if isinstance(x, tuple):
                layers.append(nn.Linear(x[0], x[1]))
            elif isinstance(x, str) and x == "R":
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
