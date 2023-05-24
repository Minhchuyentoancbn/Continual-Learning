import argparse
import sys
import random
import time
import datetime
import os
import numpy as np
import importlib
from dataloaders.mnist_permutation import get

import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--model', type=str, default='gem')

    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)

    parser.add_argument('--finetune', type=bool, default=True)

    parser.add_argument('--memory_strength', type=float, default=0.3)
    parser.add_argument('--n_memories', type=int, default=256)
    parser.add_argument('--memory_strength', type=float, default=0.5)
    parser.add_argument('--n_memories', type=int, default=256)
    # parser.add_argument('--samples_per_task', type=int, default=-1)

    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    seed_everything(args.seed)

    # Load data
    data, taskcla, size = get(args.num_tasks)

    # Load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs=size[1] * size[2], n_outputs=taskcla[0][1], n_tasks=len(taskcla), args=args)
    if args.cuda:
        model.cuda()

    # Loop tasks
    acc = np.zeros((len(taskcla), len(taskcla)))

    for t, ncla in taskcla:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        # Train
        model.train()
        for epoch in range(args.n_epochs):
            # Train
            permutation = torch.randperm(data[t]['train']['X'].size(0))
            for i in range(0, data[t]['train']['X'].size(0), args.batch_size):
                model.train()
                model.zero_grad()

                indices = permutation[i:i + args.batch_size]
                batch_x, batch_y = data[t]['train']['X'][indices], data[t]['train']['y'][indices]
                if args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                model.observe(batch_x, t, batch_y)

            # Test on validation set of current task
            model.eval()
            epoch_acc = 0
            for i in range(0, data[t]['valid']['X'].size(0), args.batch_size):
                indices = range(i, i + args.batch_size)
                batch_x, batch_y = data[t]['valid']['X'][indices], data[t]['valid']['y'][indices]
                if args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                with torch.no_grad():
                    ypred = model(batch_x, t)

                _, preds = torch.max(ypred, 1)
                epoch_acc += torch.sum(preds == batch_y).item()

            epoch_acc /= data[t]['valid']['X'].size(0)
            print('Epoch {:3d} | Accuracy {:.2f}'.format(epoch, epoch_acc))

        print('End of training for task {:d}.'.format(t))
        print('*' * 100)
        print('Computing accuracy on all tasks...')
        # Test on all tasks
        for u in range(args.num_tasks):
            model.eval()
            running_corrects = 0
            for i in range(0, data[u]['test']['X'].size(0), args.batch_size):
                indices = range(i, i + args.batch_size)
                batch_x, batch_y = data[u]['test']['X'][indices], data[u]['test']['y'][indices]
                if args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                with torch.no_grad():
                    ypred = model(batch_x, u)

                _, preds = torch.max(ypred, 1)
                running_corrects += torch.sum(preds == batch_y).item()

            acc[t, u] = running_corrects / data[u]['test']['X'].size(0)
            print('Task {:2d} | Accuracy {:.2f}'.format(u, acc[t, u]))

        print('*' * 100)
        print('Finished Task {:d}.'.format(t))
        print('*' * 100)

        # Save the results
        np.save('results/' + args.model + '_acc.npy', acc)

