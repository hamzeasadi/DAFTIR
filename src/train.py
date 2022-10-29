import conf as cfg
import model as m
import datasetup as ds
import utils
import engine
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
import wandb
from datetime import datetime
import secret
from torch.optim.lr_scheduler import ExponentialLR


my_parser = argparse.ArgumentParser(prog='train', description="training configurations!!")

my_parser.add_argument('--epoch', '-e', type=int, metavar='epoch', help='num of epochs for training')
my_parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)
my_parser.add_argument('--train', action=argparse.BooleanOptionalAction)
my_parser.add_argument('--test', action=argparse.BooleanOptionalAction)

args = my_parser.parse_args()


def run_wandb():
    dt = str(datetime.now())
    st = dt.strip().split(' ')[-1].strip().split('.')[0].strip().split(':')
    run_name = '-'.join(st) 
    wandb.login(key=secret.wandb_api_key)
    wandb.init(project='NIR DAR', name=run_name)



def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module, epochs: int=10, wandbf: bool=False):
    kt = utils.KeepTrack(path=cfg.paths['ckpoint'])
    min_eval_error = 1e+10
    model_name = f"tnn2diff.pt"
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(epochs):
        train_error = engine.train_step(net=model, opt=optimizer, data=train_loader, loss_fn=loss_fn)
        scheduler.step()
        test_error = engine.eval_step(net=model, opt=optimizer, data=test_loader, loss_fn=loss_fn)
        

        if test_error['eval_loss'] < min_eval_error:
            min_eval_error = test_error['eval_loss']
            kt.save_ckp(net=model, opt=optimizer, epoch=epoch, min_error=min_eval_error, model_name=model_name)

        
        if wandbf:
            wandb.log(dict(epoch=epoch)|train_error|test_error)

        print(dict(epoch=epoch)|train_error|test_error)
    


def main():
    wbf = args.wandb
    if wbf:
        run_wandb()

    tnn_model = m.NIRTNN2diff(dp=0.2)
    criterion = utils.NIRLoss()
    opt = utils.build_opt(Net=tnn_model, opttype='adam', lr=1e-2)
    
    if args.train:
        train_loader, test_loader = ds.build_dataloader(batch_size=10000, noise_level=1e-5, label_scale=0.1)
        train(model=tnn_model, train_loader=train_loader, test_loader=test_loader, optimizer=opt, loss_fn=criterion, epochs=args.epoch, wandbf=wbf)

    # evaluation
    if args.test:
        kt = utils.KeepTrack(path=cfg.paths['ckpoint'])
        state = kt.load_ckp(model_name=f"tnn2diff.pt")
        print(state['epoch'], state['min_error'])
        train_loader, test_loader = ds.build_dataloader(batch_size=5000, noise_level=1e-5, label_scale=0.1)
        tnn_model.load_state_dict(state_dict=state['model_state'])
        engine.test_step(net=tnn_model, data=test_loader, loss_fn=criterion)


if __name__ == '__main__':
    main()