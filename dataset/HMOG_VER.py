import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data as data


# Implement HMOG_VER for user verification
# This subset can be used for (supervised training or finetuning on ground truth labels)

class HMOG_VER(Dataset):
    def __init__(self, train_phase=False):
        super().__init__()

        self.train_train = train_phase

        if self.train_phase:
            data_frame = pd.read_excel("Auth_Train_HMOG.xlsx")
        else:
            data_frame = pd.read_excel("Auth_Test_HMOG.xlsx")

        names = ['ID', 'AGX', 'AGY', 'AGZ', 'VX', 'VY', 'VZ', 'MX', 'MY', 'MZ']
        data_frame.columns = names

        labels = data_frame['ID']

        x_frame = data_frame.drop(columns=['ID'])

        data_set = np.array(x_frame)
        data_set = data_set.astype(float)
        label_set = np.array(labels)

        rows1 = data_set.shape[0]
        cols1 = data_set.shape[1]
        num_elements1 = rows1 * cols1

        data_tensor = torch.tensor(data_set)
        label_tensor = torch.tensor(label_set)

        # 4 Implement __getitem__() and __len__()...

        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __len__(self) -> int:

        return len(self.data_tensor)

    def __getitem__(self, i):
        x = self.data_tensor[i]
        y = self.label_tensor[i]

        return (x, y)
