import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# Implement TOUCH_VER for user verification
# This subset can be used for (supervised training, or finetuning on ground truth labels)

class TOUCH_VER(Dataset):
    def __init__(self, train_phase=None):
        super().__init__()

        self.train_phase = train_phase

        if self.train_phase:
            data_frame = pd.read_excel("Auth_Train_Touch.xlsx")
        else:
            data_frame = pd.read_excel("Auth_Test_Touch.xlsx")

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

        return (x, y)
