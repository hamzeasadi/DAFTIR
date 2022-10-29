import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import torch
import shutil
from torch import nn as nn
from scipy import linalg, stats


data_path = os.path.join(os.pardir, 'data')

paths = dict(
    data=dict(
        m5_spec=os.path.join(data_path, 'corn', 'corn_m5spec_data.csv'),
        mp5_spec=os.path.join(data_path, 'corn', 'corn_mp5spec_data.csv'),
        mp6_spec=os.path.join(data_path, 'corn', 'corn_mp6spec_data.csv'),
        labels = os.path.join(data_path, 'corn', 'corn_propvals_data.csv')
    ),

    ckpoint=dict(
        ck_point=os.path.join(data_path, 'ckpoint'),
        bst_model=os.path.join(data_path, 'ckpoint', 'bst_model')
    )
)

class DataPrepare():

    def __init__(self, paths):
        self.path = paths
        
    def readData(self, dataName):
        data = pd.read_csv(self.path['data'][dataName])
        return data
    def preprocess(self, dataName):
        data = self.readData(dataName).values
        smoothed_data =  savgol_filter(x=data, window_length=11, polyorder=3)
        # scaled_data = smo
        return smoothed_data +0.1

    def allData(self):
        data_names = ['m5_spec', 'mp5_spec', 'mp6_spec']
        all_data = self.preprocess(dataName=data_names[0])
        for i in range(1, 3):
            data = self.preprocess(data_names[i])
            all_data = np.vstack((all_data, data))

        return all_data

    def to_tensor(self, dataName='', all_data=False):
        if all_data:
            data = self.allData()
        else:
            data = self.preprocess(dataName)
        
        tensor_data = torch.from_numpy(data).float().reshape(shape=(data.shape[0], 1, -1))
        # tensor_data = torch.from_numpy(data).float()
        return tensor_data

    def labels(self, label_id=0, sigmoid=False, all_data=False):

        if sigmoid:
            label = self.readData(dataName='labels').values[:, label_id]
            label = (label - min(label))/(max(label) - min(label))
            label = torch.from_numpy(label).float()
            if all_data:
                label = torch.hstack((label, label, label))
        else:
            label = self.readData(dataName='labels').values[:, label_id]
            label = torch.from_numpy(label).float()
            if all_data:
                label = torch.hstack((label, label, label))

        return label

class KeepTrack():

    def __init__(self, path=paths):
        self.state = dict(model_state_dict=(0,0), opt_state_dict=0, epoch=0, loss=0)
        self.ckpoint_path = path['ckpoint']['ck_point']
        self.bst_model_path = path['ckpoint']['bst_model']
    def ckpoint_save(self, modelName, models, opt, epoch, loss, is_bst=False):
        self.state['model_state_dict'] = (models[0].state_dict(), models[1].state_dict())
        self.state['opt_state_dict'] = opt.state_dict()
        self.state['epoch'] = epoch+1
        self.state['loss'] = loss
        torch.save(self.state, os.path.join(self.ckpoint_path, modelName))
        if is_bst:
            shutil.copyfile(os.path.join(self.ckpoint_path, modelName), os.path.join(self.bst_model_path, modelName))

    def load_model_opt(self, modelName, models, opt):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state = torch.load(os.path.join(self.bst_model_path, modelName), map_location=dev)
        models[0].load_state_dict(state['model_state_dict'][0])
        models[1].load_state_dict(state['model_state_dict'][1])
        opt.load_state_dict(state['opt_state_dict'])
        epoch = state['epoch']
        loss = state['loss']

        return models, opt, epoch, loss

    def load_model(self, modelName, models):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state = torch.load(os.path.join(self.bst_model_path, modelName), map_location=dev)
        models[0].load_state_dict(state['model_state_dict'][0])
        models[1].load_state_dict(state['model_state_dict'][1])
        # opt.load_state_dict(state['opt_state_dict'])
        epoch = state['epoch']
        loss = state['loss']

        return models, epoch, loss


class CustomeLoss(nn.Module):

    def __init__(self):
        super(CustomeLoss, self).__init__()
        self.cusmse = nn.MSELoss()

    def forward(self, src, trg, RotAlgn=True, MMD=False, SubAlgn=False):
        # error =  self.rotalng(src, trg, ssvd=True)
        # error =  self.subalng(src, trg, ssvd=True)
        # error = self.mmd(src, trg)
        error = self.subsigma(src, trg)
        return error
        

    def mmd(self, x, y, kernel='rbf'):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(dev),
                    torch.zeros(xx.shape).to(dev),
                    torch.zeros(xx.shape).to(dev))

        if kernel == "multiscale":
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1

        if kernel == "rbf":

            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)

        return torch.mean(XX + YY - 2. * XY)

    
    def subalng(self, src, trg, ssvd=False):

        if ssvd:
            Us, Ss, Vs = self.sSVD(src)
            Ut, St, Vt = self.sSVD(trg)

        else:
            Us, Ss, Vs = self.tSVD(src)
            Ut, St, Vt = self.tSVD(trg) 

        # S = torch.matmul(Us, Vs)
        # S = torch.nan_to_num(S)

        # T = torch.matmul(Ut, Vt)
        # T = torch.nan_to_num(T)

        ST = torch.matmul(Us.t(), Ut)
        ST = torch.nan_to_num(ST)

        if ssvd:
            Ps, cosSim, Pt = self.sSVD(ST)
            Ps, cosSim, Pt = torch.nan_to_num(Ps), torch.nan_to_num(cosSim), torch.nan_to_num(Pt)
        else:
           Ps, cosSim, Pt = self.tSVD(ST) 
           Ps, cosSim, Pt = torch.nan_to_num(Ps), torch.nan_to_num(cosSim), torch.nan_to_num(Pt)
        
        sinmetric = torch.sqrt(1 - torch.pow(cosSim, 2))
        sinmetric = torch.nan_to_num(sinmetric)

        sinSim = torch.nan_to_num(torch.norm(sinmetric, 1))
        bases_weight = torch.nan_to_num(torch.norm((torch.abs(Ps) - torch.abs(Pt)), 2))

        return sinSim + 0.01*bases_weight

    
    def rotalng(self, src, trg, ssvd=False):
        if ssvd:
            Us, Ss, Vs = self.sSVD(src)
            Ut, St, Vt = self.sSVD(trg)

        else:
            Us, Ss, Vs = self.tSVD(src)
            Ut, St, Vt = self.tSVD(trg) 

        S = torch.matmul(Us, Vs)
        S = torch.nan_to_num(S)

        T = torch.matmul(Ut, Vt)
        T = torch.nan_to_num(T)

        ST = torch.matmul(S.t(), T)
        ST = torch.nan_to_num(ST)

        if ssvd:
            Ps, cosSim, Pt = self.sSVD(ST)
            Ps, cosSim, Pt = torch.nan_to_num(Ps), torch.nan_to_num(cosSim), torch.nan_to_num(Pt)
        else:
           Ps, cosSim, Pt = self.tSVD(ST) 
           Ps, cosSim, Pt = torch.nan_to_num(Ps), torch.nan_to_num(cosSim), torch.nan_to_num(Pt)
        
        sinmetric = torch.sqrt(1 - torch.pow(cosSim, 2))
        sinmetric = torch.nan_to_num(sinmetric)

        sinSim = torch.nan_to_num(torch.norm(sinmetric, 1))
        bases_weight = torch.nan_to_num(torch.norm((torch.abs(Ps) - torch.abs(Pt)), 2))

        # error = sinSim + 0.01*bases_weight
        error = sinSim + self.cusmse(S, T)
        return error


    def subsigma(self, Feature_s, Feature_t):
        u_s, s_s, v_s = self.sSVD(Feature_s.t())
        u_s, s_s, v_s = torch.nan_to_num(u_s), torch.nan_to_num(s_s), torch.nan_to_num(v_s)

        u_t, s_t, v_t = self.sSVD(Feature_t.t())
        u_t, s_t, v_t = torch.nan_to_num(u_t), torch.nan_to_num(s_t), torch.nan_to_num(v_t)

        p_s, cospa, p_t = self.sSVD(torch.mm(u_s.t(), u_t))
        p_s, cospa, p_t = torch.nan_to_num(p_s), torch.nan_to_num(cospa), torch.nan_to_num(p_t)

        sinpa = torch.nan_to_num(torch.sqrt(1-torch.pow(cospa,2)))
        other = torch.nan_to_num(torch.norm(torch.abs(p_s) - torch.abs(p_t), 2))

        return torch.norm(sinpa,1)+0.001*other






    def sSVD(self, X):
        # print(X.shape)
        u, s, v = linalg.svd(X.cpu().detach().numpy(), full_matrices=False, lapack_driver="gesvd") 
        u, s, v = np.nan_to_num(u), np.nan_to_num(s), np.nan_to_num(v)
        return torch.from_numpy(u).float(), torch.from_numpy(s).float(), torch.from_numpy(v).float()

    def tSVD(self, X):
        u, s, v = torch.svd(X)
        u, s, v = torch.nan_to_num(u), torch.nan_to_num(s), torch.nan_to_num(v)
        return u, s, v
    


def orthonormal(dim=3):
    m = stats.ortho_group.rvs(dim=dim)
    return torch.from_numpy(m).float()


        

def main():
    Q = orthonormal(dim=3)
    print(Q)


if __name__ == '__main__':
    main()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import pandas as pd
import os 
import torch
from torch.utils.data import Dataset
from utils import paths, DataPrepare



class NIRDataset(Dataset):

    def __init__(self, all_data=False, path=paths, setname='', label_id=0):
        super(NIRDataset, self).__init__()
        dataset = DataPrepare(paths=paths)
        self.db = dataset.to_tensor(dataName=setname, all_data=all_data)
        self.target = dataset.labels(label_id=label_id, sigmoid=True, all_data=all_data)
    
    def __len__(self):
        return self.db.shape[0]

    def __getitem__(self, idx):
        y = self.target[idx]
        x = self.db[idx]
        return x, y

def main():
    ds = NIRDataset(all_data=True)
    print(ds[1])



if __name__ == '__main__':
    main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from torch.utils.data import DataLoader
from dataset import NIRDataset
from model import NIRNet, PLS
from torch.optim import Adam
from torch import nn as nn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from utils import KeepTrack, CustomeLoss, orthonormal
import torch
from scipy import linalg

torch.set_grad_enabled(True)

def sSVD(X):
    X = torch.nan_to_num(X)
    # u, s, v = linalg.svd(X.cpu().detach().numpy(), full_matrices=False, lapack_driver="gesvd") 
    noise = torch.randn_like(X)
    X1 = X + 1e-4*X.mean()*noise
    u, s, v = torch.linalg.svd(X1)
    u, s, v = torch.nan_to_num(u), torch.nan_to_num(s), torch.nan_to_num(v)
    return u, s, v

class Cri(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x1, x2):
        u_s, s_s, v_s = sSVD(x1.t())
        u_t, s_t, v_t = sSVD(x2.t())
        p_s, cospa, p_t = sSVD(torch.mm(u_s.t(), u_t))
        sinpa = torch.sqrt(1-torch.pow(cospa,2))
        # return torch.norm(sinpa,1)+0.01*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2) + self.criterion(s_s, s_t)
        return self.criterion(s_s, s_t)

def RSD(Feature_s, Feature_t):
    u_s, s_s, v_s = sSVD(Feature_s.t())
    u_t, s_t, v_t = sSVD(Feature_t.t())
    p_s, cospa, p_t = sSVD(torch.mm(u_s.t(), u_t))
    p_s, cospa, p_t = torch.nan_to_num(p_s), torch.nan_to_num(cospa), torch.nan_to_num(p_t)
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    return torch.nan_to_num(torch.norm(sinpa,1)+0.01*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2))


# hyperparameters
lr = 3e-4
batch_size = 80
epochs = 100000

KT = KeepTrack()

encoder = NIRNet(input_dim=700)
pls = PLS(input_dim=32)



params = list(encoder.parameters()) + list(pls.parameters())
    
opt = Adam(params=params, lr=lr)
criterion = nn.MSELoss()
customCriterion = CustomeLoss()
cr = Cri()
# (encoder, pls), _, _, _ = KT.load_model_opt(modelName='intialize.pt', models=(encoder, pls), opt=opt)



# nirdataset = NIRDataset(all_data=True, setname='', label_id=0)
# dataloader = DataLoader(dataset=nirdataset, batch_size=240, shuffle=True)

src_data = NIRDataset(all_data=False, setname='mp5_spec', label_id=0)
src_loader = DataLoader(dataset=src_data, batch_size=batch_size, shuffle=False)

trg_data = NIRDataset(all_data=False, setname='m5_spec', label_id=0)
trg_loader = DataLoader(dataset=trg_data, batch_size=batch_size, shuffle=False)

eval_loss = np.inf
# KT = KeepTrack()
modelName = 'NIRNet.pt'

for epoch in range(epochs):
    encoder.train()
    pls.train()
    for (x, y) in src_loader:
        Q1 = orthonormal(dim=batch_size)
        Q2 = orthonormal(dim=batch_size)
        x1 = torch.matmul(Q1, x.squeeze())
        x2 = torch.matmul(Q2, x.squeeze())

        z1, z2 = encoder(x2.reshape((batch_size, 1, -1)), x1.reshape((batch_size, 1, -1)))
        yhat = pls(z1.squeeze())
        
        loss = criterion(y.squeeze(), yhat.squeeze()) + RSD(z1.squeeze(), z2.squeeze())
        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch%100 == 0:
        print(f"epoch={epoch}, loss={loss.item()}")


encoder.eval()
pls.eval()
for (x, y) in trg_loader:
    # Q1 = orthonormal(dim=batch_size)
    # Q2 = orthonormal(dim=batch_size)
    # x1 = torch.matmul(Q1, x.squeeze())
    # x2 = torch.matmul(Q2, x.squeeze())

    z1, z2 = encoder(x, x)
    yhat = pls(z1.squeeze())
    
    loss = criterion(y.squeeze(), yhat.squeeze()) + cr(z1.squeeze(), z2.squeeze())
    print(f"loss={loss.item()}")
    y1 = yhat.detach().numpy()
    y = y.detach().numpy()

    print(f"r2={r2_score(y1, y)}")
    plt.scatter(y1, y)

    plt.show()

    

# for epoch in range(epochs):
#     encoder.train()
#     pls.train()
#     for (xs, ys), (xt, yt) in zip(src_loader, trg_loader):
#         # print((xs, ys), (xt, yt))
#         z1, z2 = encoder(xs, xt)
#         y1 = pls(z1.squeeze())
#         loss = criterion(y1.squeeze(), ys.squeeze()) + customCriterion(z1.squeeze(), z2.squeeze())
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
    
#     if epoch%100 == 0:
#         print(f"epoch={epoch}, loss={loss.item()}")

#     encoder.eval()
#     pls.eval()
#     with torch.no_grad():
#         loss = 0
#         for (x, y) in trg_loader:
#             z1, z2 = encoder(x, x)
#             y1 = pls(z1.squeeze())
#             loss = criterion(y1.squeeze(), y.squeeze()) 
#         if loss.item() < eval_loss:
#             is_bst = True
#             KT.ckpoint_save(modelName=modelName, models=(encoder, pls), opt=opt, epoch=epoch, loss=loss.item(), is_bst=is_bst)
#             is_bst = False
#             eval_loss = loss.item()
            



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from torch.utils.data import DataLoader
from dataset import NIRDataset
from model import NIRNet, PLS
from torch.optim import Adam
from torch import nn as nn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from utils import KeepTrack
import torch

torch.set_grad_enabled(True)

# hyperparameters
lr = 3e-4
batch_size = 80
epochs = 10000

modelName = 'NIRNet.pt'
# modelName = 'intialize.pt'
KT = KeepTrack()

encoder = NIRNet(input_dim=700)
pls = PLS(input_dim=64)
(encoder, pls), _, _ = KT.load_model(modelName=modelName, models=(encoder, pls))


src_data = NIRDataset(all_data=False, setname='mp5_spec', label_id=0)
src_loader = DataLoader(dataset=src_data, batch_size=batch_size, shuffle=False)

trg_data = NIRDataset(all_data=False, setname='m5_spec', label_id=0)
trg_loader = DataLoader(dataset=trg_data, batch_size=batch_size, shuffle=False)

encoder.eval()
pls.eval()

for (x, y) in trg_loader:
    encoder.eval(), pls.eval()
    z1, z2 = encoder(x, x)
    y1 = pls(z1.squeeze())
    y1 = y1.detach().numpy()
    y = y.detach().numpy()
    print(f"target-r2={r2_score(y1, y)}")
    plt.scatter(y1, y, label='trg')

for (x, y) in src_loader:
    encoder.eval(), pls.eval()
    z1, z2 = encoder(x, x)
    y1 = pls(z1.squeeze())
    y1 = y1.detach().numpy()
    y = y.detach().numpy()
    print(f"source-r2={r2_score(y1, y)}")
    plt.scatter(y1, y, label='src')

plt.legend()
plt.show()


# def main():
#     pass


# if __name__ == '__main__':
#     main()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import torch
from torch import nn as nn



class NIRNet(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.feat_ext = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, stride=7, padding=0), nn.ReLU(inplace=True), 
            nn.MaxPool1d(kernel_size=4, stride=2), nn.BatchNorm1d(num_features=16),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2), nn.BatchNorm1d(num_features=32),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1), 
            nn.AvgPool1d(kernel_size=3)
        )

    def forward_once(self, x):
        return self.feat_ext(x)
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


class PLS(nn.Module):

    def __init__(self, input_dim=64):
        super(PLS, self).__init__()
        self.regression = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=20), nn.ReLU(inplace=True), 
            nn.Linear(in_features=20, out_features=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.regression(x)

class ComW(nn.Module):
    def __init__(self, input_dim=80):
        super(ComW, self).__init__()
        self.fc = nn.Linear(in_features=80, out_features=80)

    def forward_once(self, x):
        return self.fc(x)

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2

def main():
    x = torch.randn(size=(80, 1, 700))
    encoder = NIRNet(input_dim=700)
    print(encoder)
    out1, out2 = encoder(x, x)
    print(f"input_dim={x.shape}, out1-dim={out1.shape}, out2-dim={out2.shape}")

    pls = PLS(input_dim=64)
    out = pls(out1.squeeze())
    print(f"out-shape={out.shape}")

if __name__ == '__main__':
    main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 