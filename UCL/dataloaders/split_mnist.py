import os
import numpy as np
import torch
from torchvision import datasets, transforms


def get(tasknum = 5):
    """
    Get Split MNIST data.

    Parameters
    ----------
    tasknum : int
        Number of tasks.

    Returns
    -------
    data : dict
        Data dictionary.

    taskcla : list
        List of tuples (task, number of classes).

    size : list
        Size of the images.
    """
    if tasknum > 5:
        tasknum = 5
    data = {}
    taskcla = []
    size = [1, 28, 28]
    
    # Pre-load
    # MNIST
    mean = (0.1307,)
    std = (0.3081,)
    if not os.path.isdir('../data/binary_split_mnist/'):
        os.makedirs('../data/binary_split_mnist')
        dat = {}
        dat['train'] = datasets.MNIST('../data/', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.MNIST('../data/', train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for i in range(5):
            data[i] = {}
            data[i]['name'] = 'split_mnist-{:d}'.format(i)
            data[i]['ncla'] = 2
            data[i]['train'] = {'X': [], 'y': []}
            data[i]['test'] = {'X': [], 'y': []}
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                task_idx = target.numpy()[0] // 2
                data[task_idx][s]['X'].append(image)
                data[task_idx][s]['y'].append(target.numpy()[0]%2)

        for i in range(5):
            for s in ['train', 'test']:
                data[i][s]['X'] = torch.stack(data[i][s]['X'])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['X'],os.path.join(os.path.expanduser('../data/binary_split_mnist'), 'data'+ str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('../data/binary_split_mnist'), 'data'+ str(i) + s + 'y.bin'))
    else:
        # Load binary files
        for i in range(5):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 2
            data[i]['name'] = 'split_mnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'X': [], 'y': []}
                data[i][s]['X'] = torch.load(os.path.join(os.path.expanduser('../data/binary_split_mnist'), 'data'+ str(i) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('../data/binary_split_mnist'), 'data'+ str(i) + s + 'y.bin'))
        
    for t in range(tasknum):
        data[t]['valid'] = {}
        data[t]['valid']['X'] = data[t]['train']['X'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    
    return data, taskcla, size