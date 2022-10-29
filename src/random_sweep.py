from importlib.metadata import distribution
import math
import torch
from torch import nn as nn, optim as optim
import wandb
import model as m
import engine 
import utils 
import datasetup as ds
import conf as cfg
import secret
from datetime import datetime


sweep_config = dict(
    method='bayes'
)

metric = dict(
    name='acc', goal='maximize'
)
sweep_config['metric'] = metric

parameters_dict = dict(
    optimizer=dict(values=['adam', 'sgd']),
    dropout=dict(values=[0.1, 0.2, 0.3, 0.4, 0.5]),
    epochs=dict(value=1),
    learning_rate= dict(distribution='uniform', min=0.0001, max=0.1),
    batch_size=dict(distribution='q_log_uniform_values', q=4, min=8, max=64)
)
sweep_config['parameters'] = parameters_dict


def train(config: dict = None):
    with wandb.init(config=config):
        config = wandb.config
        cfg.model_temp['blk4']['outch'] = config.batch_size
        model = m.NIRTNN2diff(model_temp=cfg.model_temp, dp=config.dropout)
        opt = utils.build_opt(Net=model, opttype=config.optimizer, lr=config.learning_rate)
        train_loader, test_loader = ds.build_dataloader(batch_size=config.batch_size, label_scale=0.1)
        loss_fn = utils.NIRLoss()
        train_loss = engine.train_step(net=model, opt=opt, data=train_loader, loss_fn=loss_fn)
        val_loss = engine.eval_step(net=model, opt=opt, data=test_loader, loss_fn=loss_fn)
        wandb.log({'acc': val_loss['acc']})


def run_wandb():
    dt = str(datetime.now())
    st = dt.strip().split(' ')[-1].strip().split('.')[0].strip().split(':')
    run_name = '-'.join(st) 
    wandb.login(key=secret.wandb_api_key)
    # wandb.init(project='NIR DAR', name=run_name)



def main():
    # print(sweep_config)
    dt = str(datetime.now())
    st = dt.strip().split(' ')[-1].strip().split('.')[0].strip().split(':')
    run_name = '-'.join(st)
    run_wandb()
    sweep_id = wandb.sweep(sweep=sweep_config, project=f'NIRTNN sweep-{run_name}')
    wandb.agent(sweep_id=sweep_id, function=train, count=50)


if __name__ == '__main__':
    main()