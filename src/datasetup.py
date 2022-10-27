import os
import conf as cfg
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from itertools import combinations

def create_noise_data(path: str, noise_factor:float=1e-3):
    dataframe = pd.read_csv(path)
    datanp = dataframe.values/5.0
    data_noise = datanp + noise_factor * np.random.normal(loc=0, scale=1, size=np.shape(datanp))
    noise_and_clean_data = np.vstack(tup=(datanp, data_noise))
    return noise_and_clean_data


class CornData(Dataset):
    """
    doc
    """
    def __init__(self, path, noise_level: float=1e-2, label_scale: float=1):
        super().__init__()
        self.path = path
        self.nl = noise_level
        self.ls = label_scale
        self.X_src, self.X_trg, self.Y = self.get_data()
        self.samples_idx = list(combinations(iterable=np.arange(160), r=2))

    def get_data(self):
        m5_spec_and_noise = create_noise_data(path=self.path['m5_spec'], noise_factor=self.nl)
        label0 = pd.read_csv(self.path['label0']).values/self.ls
        label = np.vstack(tup=(label0, label0))
        y = torch.from_numpy(label)
        mp5_spec_and_noise = create_noise_data(path=self.path['mp5_spec'], noise_factor=self.nl)
        X_src = torch.from_numpy(m5_spec_and_noise).unsqueeze(dim=1)
        X_trg = torch.from_numpy(mp5_spec_and_noise).unsqueeze(dim=1)
        return X_src, X_trg, y

    def __len__(self):
        return len(self.samples_idx)

    def __getitem__(self, idx):
        x_src = self.X_src[self.samples_idx[idx][0]]
        x_trg = self.X_trg[self.samples_idx[idx][1]]
        y_src = self.Y[self.samples_idx[idx][0]]
        y_trg = self.Y[self.samples_idx[idx][1]]

        return (x_src, y_src), (x_trg, y_trg)




def main():
    dataset = CornData(path=cfg.paths, noise_level=3e-3, label_scale=2)
    (x_src, y_src), (x_trg, y_trg) = dataset[10]
    print(x_src.size(), y_trg.size())
    


if __name__ == '__main__':
    main()