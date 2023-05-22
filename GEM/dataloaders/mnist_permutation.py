import numpy as np
import sys
import torch
from torchvision import datasets


def get(tasknum:int=20):
    """
    Get Permuted MNIST data.

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
    data = {}
    taskcla = []
    size = [1, 28, 28]
    # Pre-load
    # MNIST
    dat = {}
    dat['train'] = datasets.MNIST('../data/', train=True, download=True)
    dat['test'] = datasets.MNIST('../data/', train=False, download=True)
    
    for i in range(tasknum):
        print(i, end=',')
        sys.stdout.flush()
        data[i] = {}
        data[i]['name'] = 'pmnist-{:d}'.format(i)
        data[i]['ncla'] = 10
        permutation = np.random.permutation(28*28)
        for s in ['train', 'test']:
            if s == 'train':
                arr = dat[s].data.view(dat[s].data.shape[0],-1).float()
                label = torch.LongTensor(dat[s].targets)
            else:
                arr = dat[s].data.view(dat[s].data.shape[0],-1).float()
                label = torch.LongTensor(dat[s].targets)
                
            arr = arr / 255
            data[i][s]={}
            data[i][s]['X'] = arr[:, permutation].view(-1, size[0], size[1], size[2])
            data[i][s]['y'] = label
            
    # Validation
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
