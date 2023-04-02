import torch
import torch.nn as nn

import numpy as np 
import pandas as pd
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data

class HMOG_ID(Dataset): 
    
    def __init__(self, train_type, train_phase=None):
        super().__init__()
        
        
        self.train_type = train_type
        self.train_phase = train_phase
        
        
        if self.train_phase:
            data_frame = pd.read_excel("Rec_Train_HMOG.xlsx")
        else:
            data_frame = pd.read_excel("Rec_Test_HMOG.xlsx")
            
        names = ['ID', 'AGX', 'AGY', 'AGZ', 'VX', 'VY', 'VZ', 'MX', 'MY', 'MZ']
        data_frame.columns=names
        
        labels = data_frame['ID']
        
        
        x_frame = data_frame.drop(columns=['ID'])
        
        data_set = np.array(x_frame)
        data_set = data_set.astype(float) 
        label_set = np.array(labels)
        
     
        
        data_tensor = torch.tensor(data_set)
        label_tensor = torch.tensor(label_set)
    
        
        # 4 Implement __getitem__() and __len__()...
        
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        
       
    def __len__(self) -> int:
        
        return len(self.data_tensor)
    
    def __getitem__(self, i) -> torch.Tensor:
        
        x = self.data_tensor[i]
        y = self.label_tensor[i]
        
        x1 = self.augment(x)
        x2 = self.augment(x)
        
        if self.train_type == 'finetune':
            return (x,y)
        else:
            return (x1,x2)
        
    
    def augment(self, tensor, mean=0., std=1.):
        
        if self.train_type == 'pretrain':
            tensor = tensor + torch.randn(tensor.size()) * std + mean
        else:
            return tensor
        
        return tensor
