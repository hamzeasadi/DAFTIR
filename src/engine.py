import conf as cfg
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchmetrics import R2Score
import torch


def train_step(net: nn.Module, opt: optim.Optimizer, data: DataLoader, loss_fn: nn.Module):
    epoch_loss = 0.0
    net.train()
    for X, Y in data:
        predictions = net(X['x1'], X['x2'], X['x3'])
        loss = loss_fn(Y=Y, pred=predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    return dict(train_loss=epoch_loss/len(data))

def eval_step(net: nn.Module, opt: optim.Optimizer, data: DataLoader, loss_fn: nn.Module):
    epoch_loss = 0.0
    r2score = R2Score()
    r21, r23 = 0, 0
    net.eval()
    with torch.no_grad():
        for X, Y in data:
            predictions = net(X['x1'], X['x2'], X['x3'])
            loss = loss_fn(Y=Y, pred=predictions)
            epoch_loss += loss.item()
            r21 += r2score(predictions['y1'].squeeze(), Y['y1'].squeeze())
            r23 += r2score(predictions['y3'].squeeze(), Y['y3'].squeeze())
    
    l = len(data)
    return dict(eval_loss=epoch_loss/l, r21=r21/l, r23=r23/l)


def test_step(net: nn.Module, opt: optim.Optimizer, data: DataLoader, loss_fn: nn.Module):
    epoch_loss = 0.0
    r2score = R2Score()
    r21, r23 = 0, 0
    net.eval()
    with torch.no_grad():
        for X, Y in data:
            predictions = net(X['x1'], X['x2'], X['x3'])
            loss = loss_fn(Y=Y, pred=predictions)
            epoch_loss += loss.item()
            r21 += r2score(predictions['y1'].squeeze(), Y['y1'].squeeze())
            r23 += r2score(predictions['y3'].squeeze(), Y['y3'].squeeze())
    
    l = len(data)

    



def main():
    pass


if __name__ == '__main__':
    main()