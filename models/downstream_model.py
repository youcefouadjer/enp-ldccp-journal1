# Implement the downstream model for finetuning

import torch
import torch.nn as nn

class DsModel(nn.Module):
    
    def __init__(self, predmodel, num_classes):
        
        super().__init__()
        
        # predmodel = encoder (CNN) + projector (MLP)
        self.predmodel = predmodel
        self.num_classes = num_classes
        
        for p in self.predmodel.parameters():
            p.requires_grad = True
            
        for p in self.predmodel.projector.parameters():
            p.requires_grad = False
            
        self.last_layer = nn.Linear(256, self.num_classes)
        for p in self.last_layer.parameters():
            p.requires_grad = True
        
    def forward(self, x):
        out = self.predmodel.pretrained(x)
        out = self.last_layer(out)
        
        return out
