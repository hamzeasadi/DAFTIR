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
import argparse


# parser

myparser = argparse.ArgumentParser(prog='random_sweep', description="get the number of count")
myparser.add_argument('--count', '-c', metavar='count', type=int, default=100, help="number iteration for sweep")

args = myparser.parse_args()

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
    batch_size=dict(distribution='q_log_uniform_values', q=4, min=8, max=64),
    blk1_out = dict(distribution='q_log_uniform_values', q=4, min=8, max=64),
    blk2_out = dict(distribution='q_log_uniform_values', q=4, min=8, max=64),
    blk3_out = dict(distribution='q_log_uniform_values', q=4, min=8, max=64),
    blk4_out = dict(distribution='q_log_uniform_values', q=4, min=8, max=64)
)
sweep_config['parameters'] = parameters_dict


def train(config: dict = None):
    with wandb.init(config=config):
        config = wandb.config

        # cfg.model_temp['blk1']['outch'] = config.blk1_out

        # cfg.model_temp['blk2']['inch'] = config.blk1_out
        # cfg.model_temp['blk2']['outch'] = config.blk2_out

        # cfg.model_temp['blk3']['inch'] = config.blk2_out
        # cfg.model_temp['blk3']['outch'] = config.blk3_out

        # cfg.model_temp['blk4']['inch'] = config.blk3_out
        # cfg.model_temp['blk4']['outch'] = config.blk4_out

        model = m.NIRTNN2diff(model_temp=cfg.model_temp, dp=config.dropout)
        opt = utils.build_opt(Net=model, opttype=config.optimizer, lr=config.learning_rate)
        train_loader, test_loader = ds.build_dataloader(batch_size=config.batch_size, label_scale=0.1)
        loss_fn = utils.NIRLoss()
        train_loss = engine.train_step(net=model, opt=opt, data=train_loader, loss_fn=loss_fn)
        val_loss = engine.eval_step(net=model, opt=opt, data=test_loader, loss_fn=loss_fn)
        wandb.log({'acc': val_loss['acc']})


def run_sweep(sweep_configuration: dict, count: int):
    run_name = str(datetime.now()).split(' ')[-1].strip().split('.')[0].strip()
    wandb.login(key=secret.wandb_api_key)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'NIRTNN sweep-{run_name}')
    wandb.agent(sweep_id=sweep_id, function=train, count=count)



def main():
    # print(sweep_config)
    run_sweep(sweep_configuration=sweep_config, count=args.count)


if __name__ == '__main__':
    main()