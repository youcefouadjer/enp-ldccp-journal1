import torch
import torch.nn as nn


def conv1d_bn(in_channel, out_channel, stride):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU6(inplace=True)
    )
def conv_1x1_bn(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0,bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU6(inplace=True)
    )

class Inverted_Residual(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride, expand_factor):
        super(Inverted_Residual, self).__init__()
        
        self.stride= stride
        
        assert stride in [1,2]
        
        self.use_res_connect = self.stride ==1 and in_channels == out_channels
        
        self.conv = nn.Sequential(
            # point wise convolution
            
            nn.Conv1d(in_channels, in_channels * expand_factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(in_channels * expand_factor),
            nn.ReLU6(inplace=True),
            # depth wise convolution:
            
            nn.Conv1d(in_channels * expand_factor, in_channels * expand_factor, 3, stride, 
                      1, groups=in_channels * expand_factor, bias=False),
            nn.BatchNorm1d(in_channels * expand_factor),
            nn.ReLU6(inplace=True),
            # Point wise Linear:
            
            nn.Conv1d(in_channels * expand_factor, out_channels, kernel_size=1, 
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

class MobileNetV2(nn.Module):
    
    def __init__(self, batch_size, input_planes, window, num_classes, width_mult = 1.):
        super(MobileNetV2, self).__init__()
        
        # Setting the Inverted residual block
        self.batch_size = batch_size
        self.input_planes = input_planes
        self.window = window
        self.num_classes = num_classes
        
        self.inverted_residual_setting = [
            #t, c, n, s
            [1, 16, 1,1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        
        #---The first Layer---
        
        assert self.window % 32 == 0
        input_channel = int(32 * width_mult)
        
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        
        self.features = [conv1d_bn(input_planes, input_channel, stride=2)]
        
        # Inverted Residual Blocks:
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = int(c * width_mult)
            
            for i in range(n):
                if i==0:
                    self.features.append(Inverted_Residual(input_channel, output_channel, s, t))
                else:
                    self.features.append(Inverted_Residual(input_channel, output_channel, 1, t))
                    
                input_channel = output_channel
                
        # Last Layers:
        
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool1d(8))
        
        # Transform features ---> to nn.Sequential
        self.features = nn.Sequential(*self.features)
        
        # Lastly the classifier:
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, self.num_classes)
        )
        self._initialize_weights()
        
    def forward(self, x):
        # reshape the input:
        x = x.reshape(self.batch_size, self.input_planes, self.window)
        x = self.features(x)
        # reshape again for classification:
        
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        
        return x
    
    
    def _initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
