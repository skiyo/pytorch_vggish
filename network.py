
import torch
import torch.nn as nn

def make_layers(cfg, batch_norm=False, in_channels=16):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGGishModel(nn.Module):

    def __init__(self, features, num_classes=1024):
        super(VGGishModel, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1, 96, 64)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vggish(**kwargs):
    vggish_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']
    return VGGishModel(make_layers(vggish_cfg), **kwargs)

