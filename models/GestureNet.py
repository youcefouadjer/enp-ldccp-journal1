import torch
import torch.nn as nn

# Build The GestureNet Model:

class GestureNet(nn.Module):
    
    def __init__(self, batch_size, input_planes, window, num_classes):
        super().__init__()
        
        self.batch_size = batch_size
        self.input_planes = input_planes
        self.window = window
        self.num_classes = num_classes
        
        def depthwise_conv(in_ch, out_ch, stride=1):
            
            return nn.Sequential(nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_ch),
            nn.ReLU())
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(self.input_planes, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            depthwise_conv(in_ch=32, out_ch=64, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        
        self.layer3 = nn.Sequential(
            depthwise_conv(in_ch=64, out_ch=128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            depthwise_conv(in_ch=128, out_ch=256, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AvgPool1d(8)
        
        self.fc = nn.Linear(256, self.num_classes)
        
    def forward(self, x):
        # Reshape The input
        x = x.reshape(self.batch_size, self.input_planes, self.window)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc(x)
       
        x = self.dropout(x)
        
        return x
    
        
        
        
