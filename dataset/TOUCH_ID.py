import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TOUCH_ID(Dataset):
    def __init__(self, train_type, train_phase=None):
        super().__init__()

        self.train_type = train_type
        self.train_phase = train_phase

        if self.train_phase:
            data_frame = pd.read_excel("Rec_Train_Touch.xlsx")
        else:
            data_frame = pd.read_excel("Rec_Test_Touch.xlsx")

        names = ['user ID', 'action', 'phone orientation',
                 'x-coord', 'y-coord', 'pressure', 'area covered', 'finger orient']
        data_frame.columns = names

        labels = data_frame['user ID']

        x_frame = data_frame.drop(columns=['user ID'])

        data_set = np.array(x_frame)
        data_set = data_set.astype(float)
        label_set = np.array(labels)

        data_tensor = torch.tensor(data_set)
        label_tensor = torch.tensor(label_set)

        # 4 Implement __getitem__() and __len__()...

        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, i):
        x = self.data_tensor[i]
        y = self.label_tensor[i]

        x1 = self.augment(x)
        x2 = self.augment(x)

        if self.train_type == 'finetune':
            return (x, y)
        else:
            return (x1, x2)

    def augment(self, tensor, mean=0., std=1.):

        if self.train_type == 'pretrain':
            tensor = tensor + torch.randn(tensor.size()) * std + mean
        else:
            return tensor

        return tensor
