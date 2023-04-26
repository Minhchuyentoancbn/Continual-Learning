import numpy as np
import matplotlib.pyplot as plt
import time
import random

import torch
import torch.nn as nn

from utils import *
from dataloaders import row_pmnist
from approaches import ewc, si, ucl, vcl, hat
from networks import mlp_ucl, mlp, mlp_hat

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(42)

split = False  # Single-head
notMNIST = False

# Load data
print('Load data...')
data, taskcla, inputsize = row_pmnist.get(tasknum=10)
print('Input size =', inputsize, '\nTask info =', taskcla)


hat_args = {
    'experiment': 'row_pmnist',
    'approach': 'hat',
    'alpha': 0.75,
    'smax': 400,
    'seed': 42,
    'lr': 0.05,
    'units': 400,
    'batch_size': 256,
    'epochs': 100,
    'optimizer': 'sgd',
    'tasknum': 10,
    'parameter': '',
    'conv_net': False
}

log_name = '{}_{}_{}_alpha_{}_smax_{}_lr_{}_units_{}_batch_{}_epoch_{}'.format(
        hat_args['experiment'], hat_args['approach'], hat_args['seed'], hat_args['alpha'], hat_args['smax'],
        hat_args['lr'], hat_args['units'], hat_args['batch_size'], hat_args['epochs'])

hat_args['output'] = './result_data/' + log_name + '.txt'


torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = mlp_hat.Net(inputsize, taskcla, hat_args['units'], split, notMNIST).cuda()
approach = hat.Approach(net, 
                        hat_args['epochs'],
                        hat_args['batch_size'],
                        hat_args['lr'],
                        args=hat_args,
                        log_name=log_name,
                        split=split)

print_model_report(net)
print_optimizer_config(approach.optimizer)


acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
loss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

for t, ncla in taskcla:
    if t == hat_args['tasknum']:
        break

    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Load data
    X_train = data[t]['train']['X'].cuda()
    y_train = data[t]['train']['y'].cuda()
    X_valid = data[t]['valid']['X'].cuda()
    y_valid = data[t]['valid']['y'].cuda()
    task = t

    # Train
    approach.train(task, X_train, y_train, X_valid, y_valid, data)
    print('-' * 100)

    # Test
    for u in range(t + 1):
        X_test = data[u]['test']['X'].cuda()
        y_test = data[u]['test']['y'].cuda()
        test_loss, test_acc = approach.eval(u, X_test, y_test)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                      100 * test_acc))
        acc[t, u] = test_acc
        loss[t, u] = test_loss

    # Save
    print('Save at ' + hat_args['output'])
    np.savetxt(hat_args['output'], acc, fmt='%.4f')
    torch.save(net.state_dict(), './models/trained_model/'+ log_name + '_task_{}.pt'.format(t))

# Done
print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
print('Done!')