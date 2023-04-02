import torch
import torch.nn as nn 
from math import ceil

import torch
import torch.nn as nn
from math import ceil
base_model = [
    #t, c, n, s, k
    [1, 16, 1, 1, 3],
    [6, 24, 2, 1, 3],
    [6, 40, 2, 1, 3],
    [6, 80, 3, 1, 3],
    [6, 112, 3, 1, 3],
    [6, 192, 4, 1, 3], 
    [6, 320, 1, 2, 3]
]

phi_values = {
    # tuple of values: (phi_value, resolution, drop_rate)
    "b0": (0, 32, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)
}

class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1,
                bias=False):
        super(CNN_Block, self).__init__()
        
        self.cnn = nn.Conv1d(in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            groups=groups)
        self.bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU() #SiLU <--> Swish
        
    def forward(self, x):
        
        return self.silu(self.bn(self.cnn(x)))


class Squeeze_Ext(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(Squeeze_Ext, self).__init__()
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Lentgth x n°Planes --> N°Planes x 1
            nn.Conv1d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid()   
        )
        
    def forward(self, x):
        return x * self.se(x)


class Inverted_Residual(nn.Module):
    
    
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 expand_ratio,
                 reduction = 4, # -> for squeez excitation block
                 survival_prob=0.8):
        
        super(Inverted_Residual, self).__init__()
        
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels/reduction)
        
        if self.expand:
            self.expand_conv = CNN_Block(in_channels, hidden_dim, kernel_size=3, 
                                    stride=1, padding=1)
        self.conv = nn.Sequential(
            CNN_Block(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                      padding=padding, groups=hidden_dim),
            
            Squeeze_Ext(hidden_dim, reduced_dim),
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False))
    
       
    
    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs 
        
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)
        
        

class EfficientNet(nn.Module):
    def __init__(self, batch_size, input_planes, window, num_classes, version='b0'):
        
        super(EfficientNet, self).__init__()
        
        width_factor, depth_factor, drop_rate = self.calculate_factors(version)
        
        self.batch_size = batch_size
        self.input_planes = input_planes
        self.window = window
        self.num_classes = num_classes
        
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(last_channels, self.num_classes)
        )
        
        
    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNN_Block(self.input_planes, channels, 3, stride=2, padding=1)]
        in_channels = channels
        
        for t, c, n, s, k in base_model:
            out_channels = 4 * ceil(int(channels * width_factor)/4)
            layers_repeats = ceil(n * depth_factor)
            
            for layer in range(layers_repeats):
                features.append(Inverted_Residual(
                    in_channels,out_channels,
                    kernel_size=k,
                    stride = s if layer == 0 else 1,
                    padding=1,
                    expand_ratio=t)
                               )
                
                in_channels = out_channels
        features.append(
            CNN_Block(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )
        
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = x.reshape(self.batch_size, self.input_planes, self.window)
        x = self.pool(self.features(x))
        x = x.reshape(x.shape[0], -1)
        
        return self.classifier(x)
                
