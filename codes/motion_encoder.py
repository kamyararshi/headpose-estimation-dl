import torch
import torch.nn as nn
import torchvision.models as models

"""
class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet18Sub(nn.Module):
    def __init__(self, in_channels, pretrained=False):
        super(ResNet18Sub, self).__init__()
        self.pretrained = pretrained

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512, num_classes)

        if self.pretrained:
            self.model = models.resnet18(weights=True)


    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or planes != 64:
            downsample = nn.Sequential(
                nn.Conv2d(64, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(ResBlock(64, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(ResBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.pretrained:
            return self.model(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            #x = x.view(x.size(0), -1)
            #x = self.fc(x)

            return x
"""        
class ResNet18(nn.Module):
    def __init__(self, in_channels, pretrained=True):
        super(ResNet18, self).__init__()
        # If we want pretrained weights or not
        if pretrained:
            self.model = models.resnet18(weights=pretrained)
        else:
            self.model = models.resnet18(weights=False)
        # in_channels    
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Set last fully connected layer to output 7 numbers;
        # 3 for rotation, 3 for translation and 1 for scale (also needed to build translation matrix afterwards)
        self.model.fc = nn.Linear(512, 7)

    def forward(self, x):
        return self.model(x)
