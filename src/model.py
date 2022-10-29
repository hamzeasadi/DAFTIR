import conf as cfg
import torch
from torch import nn as nn, functional as F
from torchinfo import summary

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create model
class NIRTNN2diff(nn.Module):
    """
    doc
    """
    def __init__(self, model_temp: dict=cfg.model_temp, dp: float=0.1):
        super(NIRTNN2diff, self).__init__()
        self.temp = model_temp
        self.dp = dp
        keys = list(model_temp.keys())
        self.fxx = self.fx(temp=model_temp)
        self.reg = nn.Sequential(nn.Flatten(), nn.Linear(in_features=model_temp[keys[-1]]['outch'], out_features=1))
        self.regdiff = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=2*model_temp[keys[-1]]['outch'], out_features=model_temp[keys[-1]]['outch']),
            nn.Tanh(), nn.Linear(in_features=model_temp[keys[-1]]['outch'], out_features=1)
            )

    def _blk(self, blk: dict):
        layer = nn.Sequential(
            nn.Conv1d(in_channels=blk['inch'], out_channels=blk['outch'], kernel_size=blk['ks'], stride=blk['stride']),
            # nn.Conv1d(in_channels=blk['inch'], out_channels=blk['outch'], kernel_size=1),
            nn.BatchNorm1d(num_features=blk['outch']), nn.LeakyReLU(negative_slope=0.2)
        )
        if blk['pool']:
            layer = nn.Sequential(layer, nn.AvgPool1d(kernel_size=3))
        
        if blk['dropout']:
            layer = nn.Sequential(layer, nn.Dropout1d(p=self.dp))

        return layer

    def fx(self, temp: dict):
        modules = [nn.Sequential(self._blk(blk=temp[key])) for key in temp.keys()]
        feature_xt = nn.Sequential(*modules)
        return feature_xt

    def forward_once(self, x):
        return self.fxx(x)

    def forward(self, x1, x2, x3):
        fx1 = self.forward_once(x1)
        fx2 = self.forward_once(x2)
        fx3 = self.forward_once(x3)
        
        noise = 3e-4 * torch.randn_like(fx1, device=dev)
        fx1n = fx1 + noise
        fx2n = fx2 + noise
        fx3n = fx3 + noise
        
        y1 = self.reg(fx1)
        y3 = self.reg(fx3)

        # fx12 = torch.concat(tensors=(fx1, fx2), dim=1)
        # fx23 = torch.concat(tensors=(fx2, fx3), dim=1)
        # fx31 = torch.concat(tensors=(fx3, fx1), dim=1)

        # y12 = self.regdiff(fx12)
        # y23 = self.regdiff(fx23)
        # y31 = self.regdiff(fx31)


        fx12n = torch.concat(tensors=(fx1n, fx2n), dim=1)
        fx23n = torch.concat(tensors=(fx2n, fx3n), dim=1)
        fx31n = torch.concat(tensors=(fx3n, fx1n), dim=1)

        y12 = self.regdiff(fx12n)
        y23 = self.regdiff(fx23n)
        y31 = self.regdiff(fx31n)

        return dict(y12=y12, y23=y23, y31=y31, y1=y1, y3=y3, z1=fx1, z2=fx2, z3=fx3)
        # return y12, y23, y31, y1, y3, fx1, fx2, fx3





def main():
    model = NIRTNN2diff(model_temp=cfg.hyper['model'], dp=0.2)
    # summary(model, input_size=[(10, 3, 140), (10, 3, 140), (10, 3, 140)])
    x1 = torch.randn(size=(32, 3, 140))
    x2 = torch.randn(size=(32, 3, 140))
    out = model(x1, x2, x1)
    print(out['y12'].shape, out['z1'].shape)
   



if __name__ == '__main__':
    main()

