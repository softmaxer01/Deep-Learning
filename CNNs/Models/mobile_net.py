import torch
import torch.nn as nn


class DepthWiseConv(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, 3, stride, 1, groups=channels, bias=False
        )

    def forward(self, x):
        return self.conv(x)


class MobileNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.conv_layer = self._create_conv(arch["conv"])
        self.linear_layer = self._create_linear(arch["linear"])

    def forward(self, x):
        return self.linear_layer(self.conv_layer(x))

    def _create_conv(self, conv_arch):
        layers = []
        for cfg in conv_arch:
            if isinstance(cfg, tuple):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(cfg[0], cfg[1], cfg[2], cfg[3], bias=False),
                        nn.BatchNorm2d(cfg[1]),
                        nn.ReLU(inplace=True),
                    )
                )
            elif isinstance(cfg, list):
                op = cfg[0]
                if op == "dw":
                    layers.append(
                        nn.Sequential(
                            DepthWiseConv(cfg[1], cfg[2]),
                            nn.BatchNorm2d(cfg[1]),
                            nn.ReLU(inplace=True),
                        )
                    )
                elif op == "pw":
                    layers.append(
                        nn.Sequential(
                            nn.Conv2d(cfg[1], cfg[2], 1, bias=False),
                            nn.BatchNorm2d(cfg[2]),
                            nn.ReLU(inplace=True),
                        )
                    )
                elif op == "avgpool":
                    layers.append(nn.AdaptiveAvgPool2d((cfg[1], cfg[2])))
        return nn.Sequential(*layers)

    def _create_linear(self, linear_arch):
        layers = nn.ModuleList([nn.Flatten()])
        for cfg in linear_arch:
            if isinstance(cfg, tuple):
                layers.append(nn.Linear(cfg[0], cfg[1]))
        return nn.Sequential(*layers)    

def mobile_net_arch():
    return {
        "conv": [
            (3, 32, 3, 2),
            ["dw", 32, 1],
            ["pw", 32, 64],
            ["dw", 64, 2],
            ["pw", 64, 128],
            ["dw", 128, 1],
            ["pw", 128, 128],
            ["dw", 128, 2],
            ["pw", 128, 256],
            ["dw", 256, 1],
            ["pw", 256, 256],
            ["dw", 256, 2],
            ["pw", 256, 512],
            ["dw", 512, 1],
            ["pw", 512, 512],
            ["dw", 512, 1],
            ["pw", 512, 512],
            ["dw", 512, 1],
            ["pw", 512, 512],
            ["dw", 512, 1],
            ["pw", 512, 512],
            ["dw", 512, 1],
            ["pw", 512, 512],
            ["dw", 512, 2],
            ["pw", 512, 1024],
            ["dw", 1024, 2],
            ["pw", 1024, 1024],
            ["avgpool", 1, 1],
        ],
        "linear": [(1024, 10)],
    }
# arch = mobile_net_arch()
# model = MobileNet(arch=arch)
# img = torch.randn((1,3,224,224))
# print(model(img).shape)


# total_params = sum(param.numel() for param in model.parameters())
# print(f"Total number of parameters: {total_params}")


