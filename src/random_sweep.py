import math
import torch
from torch import nn as nn, optim as optim
import wandb


sweep_config = dict(
    method='random'
)

metric = dict(
    name='acc', goal='maximize'
)
sweep_config['metric'] = metric






def main():
    print(sweep_config)



if __name__ == '__main__':
    main()