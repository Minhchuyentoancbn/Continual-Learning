import numpy as np
import random
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
from dataloaders.base import MNIST
from dataloaders.datasetGen import SplitGen
from agents.regularization import SI

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)
random.seed(42)
np.random.seed(42)

train_dataset, val_dataset = MNIST(r'../data/')
train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset, first_split_size=2, other_split_size=2, rand_split=False, remap_class=True)

# Hyperparameters
# Data params
input_dim = 784
output_dim = 10

# Network params
n_hidden_units = 256

# Optimization params
batch_size = 64
epochs_per_task = 10

# Reset optimizer after each age
reset_optimizer = True

agent_config = {
    'epochs': epochs_per_task,
    'lr': 1e-3,
    'weight_decay': 0,
    'model_type': 'mlp',
    'model_name': 'MLP256_MNIST',
    'out_dim': task_output_space,
    'model_weights': None,
    'print_freq': 1,
    'gpu': True
}

agent = SI(agent_config)


