import torch
import torch.nn as nn

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class CascadedFCN(nn.Module):

    def __init__(self, features):
        super(CascadedFCN, self).__init__()
        self.features = features

        # Convolution layers replacing the fully connected layers
        self.conv1 = nn.Conv2d(512, 4096, kernel_size=7)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout()

        self.conv2 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout()

        self.conv3 = nn.Conv2d(4096, 21, kernel_size=1)  # assuming 21 classes for segmentation

        # Transpose convolution layers for upsampling
        self.upconv1 = nn.ConvTranspose2d(21, 21, kernel_size=64, stride=32, padding=16, bias=False)

    def forward(self, x):
        x = self.features(x)

        # print(f"Features output size: {x.size()}")
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.conv3(x)

        x = self.upconv1(x)

        return x


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(input_channel, l, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(l), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            input_channel = l
    return nn.Sequential(*layers)


def cascaded_fcn():
    return CascadedFCN(make_layers(cfg['D'], batch_norm=True))

