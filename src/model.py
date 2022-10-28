import conf as cfg
import torch
from torch import nn as nn, functional as F
from torchinfo import summary


# create model
class NIRTNN2diff(nn.Module):
    """
    doc
    """
    def __init__(self, model_temp: dict=cfg.model_temp):
        super(NIRTNN2diff, self).__init__()
        self.temp = model_temp
        self.blk1 = self._blk(blk=model_temp['blk1'])
        self.blk2 = self._blk(blk=model_temp['blk2'])
        self.blk3 = self._blk(blk=model_temp['blk3'])
        self.blk4 = self._blk(blk=model_temp['blk4'])
        self.blk5 = self._blk(blk=model_temp['blk5'])

        self.reg = nn.Sequential(nn.Flatten(), nn.Linear(in_features=model_temp['blk5']['outch'], out_features=1))
        self.regdiff = nn.Sequential(nn.Flatten(), nn.Linear(in_features=2*model_temp['blk5']['outch'], out_features=1))

    def _blk(self, blk: dict):

        layer = nn.Sequential(
            nn.Conv1d(in_channels=blk['inch'], out_channels=blk['outch'], kernel_size=blk['ks'], stride=blk['stride']),
            nn.BatchNorm1d(num_features=blk['outch']), nn.LeakyReLU(negative_slope=0.2)
        )
        if blk['pool']:
            layer = nn.Sequential(layer, nn.AvgPool1d(kernel_size=3))
        
        if blk['dropout']:
            layer = nn.Sequential(layer, nn.Dropout1d(p=0.2))

        return layer

    def forward_once(self, x):
        return self.blk5(self.blk4(self.blk3(self.blk2(self.blk1(x)))))

    def forward(self, x1, x2, x3):
        fx1 = self.forward_once(x1)
        fx2 = self.forward_once(x2)
        fx3 = self.forward_once(x3)
        
        y1 = self.reg(fx1)
        y3 = self.reg(fx3)

        fx12 = torch.concat(tensors=(fx1, fx2), dim=1)
        fx23 = torch.concat(tensors=(fx2, fx3), dim=1)
        fx31 = torch.concat(tensors=(fx3, fx1), dim=1)

        y12 = self.regdiff(fx12)
        y23 = self.regdiff(fx23)
        y31 = self.regdiff(fx31)

        return dict(y12=y12, y23=y23, y31=y31, y1=y1, y3=y3, z1=fx1, z2=fx2, z3=fx3)





def main():
    model = NIRTNN2diff()
    # summary(model, input_size=[(10, 1, 140), (10, 1, 140)])
    x1 = torch.randn(size=(32, 1, 140))
    x2 = torch.randn(size=(32, 1, 140))
    out = model(x1, x2, x1)
    print(out['y12'].shape, out['z1'].shape)
   



if __name__ == '__main__':
    main()

