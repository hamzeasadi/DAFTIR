import os
import conf as cfg
import torch
from torch import nn as nn, optim


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_opt(Net: nn.Module, opttype: str='adam', lr: float=9e-4):
    if opttype == 'adam':
        opt = optim.Adam(params=Net.parameters(), lr=lr)
    elif opttype == 'sgd':
        opt = optim.SGD(params=Net.parameters(), lr=lr)

    return opt


class NIRLoss(nn.Module):
    """
    doc
    """
    def __init__(self) -> None:
        super().__init__()
        self.crt = nn.SmoothL1Loss()

    def RSD(self, Feature_s, Feature_t, tradeoff2):
        u_s, s_s, v_s = torch.svd(Feature_s.t())
        u_t, s_t, v_t = torch.svd(Feature_t.t())
        p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
        sinpa = torch.sqrt(1-torch.pow(cospa,2))

        return torch.norm(sinpa,1)+ tradeoff2*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)

    def MMD(self, x, y, kernel):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
        
        XX, YY, XY = (torch.zeros(xx.shape),
                    torch.zeros(xx.shape),
                    torch.zeros(xx.shape))
        
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


    def forward(self, Y: dict, pred: dict):
        # loss1 = self.crt(pred['y12'].squeeze(), (Y['y1'] - Y['y2']).squeeze().to(dev))
        # loss2 = self.crt(pred['y23'].squeeze(), (Y['y2'] - Y['y3']).squeeze().to(dev))
        # loss3 = self.crt(pred['y31'].squeeze(), (Y['y3'] - Y['y1']).squeeze().to(dev))
        # loss4 = self.crt(pred['y1'].squeeze(), Y['y1'].squeeze().to(dev))
        # loss5 = self.crt(pred['y3'].squeeze(), Y['y3'].squeeze().to(dev))

        y_zero = pred['y12'] + pred['y23'] + pred['y31']
        yhat1 = Y['y3'] - pred['y31']
        yhat3 = pred['y31'] + Y['y1']
    
        loss1 = self.crt(y_zero, torch.zeros_like(y_zero))
        loss2 = self.crt(yhat1.squeeze(), Y['y1'].squeeze())
        loss3 = self.crt(yhat3.squeeze(), Y['y3'].squeeze())

        loss4 = self.crt(pred['y1'].squeeze(), Y['y1'].squeeze())
        loss5 = self.crt(pred['y3'].squeeze(), Y['y3'].squeeze())
        
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        return loss
        

class KeepTrack():
    """
    doc
    """
    def __init__(self, path: str) -> None:
        self.state = dict(model_state='', opt_state='', min_error=0.0, epoch=0)
        self.path = path
    
    def save_ckp(self, net: nn.Module, opt: optim.Optimizer, epoch: int, min_error: float, model_name: str):
        self.state['model_state'] = net.state_dict()
        self.state['opt_state'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['min_error'] = min_error
        save_path = os.path.join(self.path, model_name)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, model_name: str):
        return torch.load(f=os.path.join(self.path, model_name))

def main():
    t1 = torch.randn(size=(3, 3), requires_grad=True)
    t2 = t1.clone().detach()
    t2sign = torch.sign(t2)
    x = torch.randn(size=(3, 3))
    i = torch.eye(n=x.size(0))
    xi = x+i
    ux, sx, vx = torch.svd(x)
    # uxi, sxi, vxi = torch.svd(xi)
    # print(ux, sx, vx)
    # print(uxi, sxi, vxi)
    # print(ux@ux.t(), ux.t()@ux)
    t3 = torch.randn_like(t1)
    print(t3.requires_grad)
    

    
    


if __name__ == '__main__':
    main()