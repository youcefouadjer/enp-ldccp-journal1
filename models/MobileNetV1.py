import torch
import torch.nn as nn


class MobileNetV1(nn.Module):
    def __init__(self, batch_size, input_planes, window, num_classes, layers=[2, 2, 6, 1]):
        super().__init__()
        
        
        self.batch_size = batch_size
        self.input_planes = input_planes
        self.window = window
        self.num_classes = num_classes
        
        self.in_channels = 32
        
        def depthwise_conv(in_ch, out_ch, stride=1):
            
            return nn.Sequential(nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_ch),
            nn.ReLU())
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.input_planes, self.in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU())
        
        self.layer0 = depthwise_conv(self.in_channels, 64)
        
        self.layer1 = self._create_layer(depthwise_conv, num_layers=layers[0], out_channels=128, stride=1) # 2
        self.layer2 = self._create_layer(depthwise_conv, num_layers=layers[1], out_channels=256, stride=1) # 2
        self.layer3 = self._create_layer(depthwise_conv, num_layers=layers[2], out_channels=512, stride=1) # 6
        self.layer4 = self._create_layer(depthwise_conv, num_layers=layers[3], out_channels=1024, stride=2)# 1
        
        
        self.layer5 = depthwise_conv(1024, 1024)
        
        self.avgpool = nn.AvgPool1d(8)
        self.fc = nn.Linear(1024, self.num_classes)
        
    def forward(self, x):
        # reshaping the input
        x = x.view(self.batch_size, self.input_planes, self.window)
        x = self.conv1(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.avgpool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    
        
    def _create_layer(self, depthwise_conv, num_layers, out_channels, stride):
        
        layers = []
        
        layers.append(depthwise_conv(int(out_channels//2), out_channels, stride=stride))
        self.in_channels = out_channels
        
        for i in range(0, num_layers - 1):
            layers.append(depthwise_conv(self.in_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)
    
