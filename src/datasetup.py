import os
import conf as cfg
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from itertools import combinations
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

def scl(x):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(x)
    return scaler.transform(x)

def create_noise_data(path: str, noise_factor:float=1e-3):
    dataframe = pd.read_csv(path)
    datanp = dataframe.values/5.0
    data_noise = datanp + noise_factor * np.random.normal(loc=0, scale=1, size=np.shape(datanp))
    noise_and_clean_data = np.vstack(tup=(datanp, data_noise))
    return noise_and_clean_data
    # return datanp

def create_noise_data1(path: str):
    dataframe = pd.read_csv(path)
    # dataframe2 = pd.read_csv(cfg.paths['mp6_spec'])
    datanp = dataframe.values/5.0
    # dnp = dataframe2.values/5.0
    # noise_and_clean_data = np.vstack(tup=(datanp, dnp))
    return datanp

def create_drive_data(path: str):
    dataframe = pd.read_csv(path)
    datanp = scl(dataframe.values/5.0)

    data_drive = scl(savgol_filter(x=datanp, window_length=7, polyorder=3, deriv=1))
    data_drive2 = scl(savgol_filter(x=datanp, window_length=7, polyorder=3, deriv=2))

    datanp_expand = np.expand_dims(datanp, axis=1)
    data_drive_expand = np.expand_dims(data_drive, axis=1)
    # data_drive2_expand = np.expand_dims(data_drive2, axis=1)

    data = np.concatenate((datanp_expand, data_drive_expand), axis=1)
    # return noise_and_clean_data
    return data

class CornData(Dataset):
    """
    doc
    """
    def __init__(self, path, noise_level: float=1e-2, label_scale: float=0.1):
        super().__init__()
        self.path = path
        self.nl = noise_level
        self.ls = label_scale
        self.X_src, self.X_trg, self.Y = self.get_data()
        self.samples_idx = list(combinations(iterable=np.arange(80), r=3))

    def get_data(self):
        # m5_spec_and_noise = create_noise_data1(path=self.path['m5_spec'])
        m5_spec_and_noise = create_drive_data(path=self.path['m5_spec'])
        label0 = pd.read_csv(self.path['label0']).values
        # label = np.vstack(tup=(label0, label0))
        y = torch.from_numpy(label0)
        y = y - y.min() + self.ls
        # mp5_spec_and_noise = create_noise_data1(path=self.path['mp5_spec'])
        mp5_spec_and_noise = create_drive_data(path=self.path['mp5_spec'])
        X_src = torch.from_numpy(m5_spec_and_noise)
        X_trg = torch.from_numpy(mp5_spec_and_noise)
        return X_src.type(torch.float32), X_trg.type(torch.float32), y.type(torch.float32)

    def __len__(self):
        return len(self.samples_idx)

    def __getitem__(self, idx):
        x1 = self.X_src[self.samples_idx[idx][0]]
        x2 = self.X_trg[self.samples_idx[idx][1]]
        x3 = self.X_src[self.samples_idx[idx][2]]
        y1 = self.Y[self.samples_idx[idx][0]]
        y2 = self.Y[self.samples_idx[idx][1]]
        y3 = self.Y[self.samples_idx[idx][2]]
        return dict(x1=x1, x2=x2, x3=x3), dict(y1=y1, y2=y2, y3=y3)




class NIRData(Dataset):
    """
    doc
    """
    def __init__(self, path, noise_level: float=1e-2, label_scale: float=0.1):
        super().__init__()
        self.path = path
        self.nl = noise_level
        self.ls = label_scale
        self.X_src, self.X_trg, self.Y = self.get_data()
        self.samples_idx = list(combinations(iterable=np.arange(80), r=3))

    def get_data(self):
        # m5_spec_and_noise = create_noise_data1(path=self.path['m5_spec'])
        m5_spec_and_noise = create_drive_data(path=self.path['m5_spec'])
        label0 = pd.read_csv(self.path['label0']).values
        # label = np.vstack(tup=(label0, label0))
        y = torch.from_numpy(label0)
        y = y - y.min() + self.ls
        # mp5_spec_and_noise = create_noise_data1(path=self.path['mp5_spec'])
        mp5_spec_and_noise = create_drive_data(path=self.path['mp5_spec'])
        X_src = torch.from_numpy(m5_spec_and_noise)
        X_trg = torch.from_numpy(mp5_spec_and_noise)
        return X_src.type(torch.float32), X_trg.type(torch.float32), y.type(torch.float32)

    def sample(self, batch_size: int=32):
        x = np.arange(80)
        xs = np.random.choice(a=x, size=3*batch_size, replace=False)
        xs1, xt, xs2 = np.split(xs, 3)
        return xs1, xt, xs2

    def get_batch(self, batch_size: int=16):
        xs1_idx, xt_idx, xs2_idx = self.sample(batch_size=batch_size)
        Xs1, Xt, Xs2 = self.X_src[xs1_idx], self.X_trg[xt_idx], self.X_src[xs2_idx]
        ys1, yt, ys2 = self.Y[xs1_idx], self.Y[xt_idx], self.Y[xs2_idx]

        return dict(x1=Xs1, x2=Xt, x3=Xs2), dict(y1=ys1, y2=yt, y3=ys2)



def build_dataloader(batch_size: int=32, noise_level: float=1e-3, label_scale:float=1):
    dataset = CornData(path=cfg.paths, noise_level=noise_level, label_scale=label_scale)
    l = len(dataset)
    train_size = int(l*0.8)
    test_size = l - train_size
    train_set, test_set = random_split(dataset=dataset, lengths=[train_size, test_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)

    return train_loader, test_loader



def main():
    # dataset = CornData(path=cfg.paths, noise_level=3e-3, label_scale=0)
    dataset = NIRData(path=cfg.paths, noise_level=0.001, label_scale=0.0)
    X, Y = dataset.get_batch(batch_size=10)
    print(X['x1'].shape, Y['y2'].shape)
    # data = create_drive_data(path=cfg.paths['m5_spec'])

    # for i in range(2):
    #     plt.plot(data[0][i])
    
    # plt.show()
    


if __name__ == '__main__':
    main()