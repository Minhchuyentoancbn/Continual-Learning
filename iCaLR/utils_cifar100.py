import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR100
from torchvision import transforms


def load_data(samples_per_class_val: int, seed: int = 42):
    """
    Load the CIFAR-100 dataset and split it into tasks.

    Parameters
    ----------
    samples_per_class_val : int
        Number of validation samples per class.

    seed : int
        Random seed.
    """
    np.random.seed(seed)

    cifar100_train = CIFAR100(root='../data/', train=True, download=True)
    cifar100_test = CIFAR100(root='../data/', train=False, download=True)

    # Normalize the data per pixels
    X = torch.cat([torch.from_numpy(cifar100_train.data), torch.from_numpy(cifar100_test.data)], dim=0).permute(0, 3, 1, 2).float() / 255.0
    X = X -  X[:50000].mean(dim=0)

    # Create Train/Validation set
    eff_samples_class = 500 - samples_per_class_val
    X_train = torch.zeros((100 * eff_samples_class, 3, 32, 32))
    y_train = torch.zeros((100 * eff_samples_class, ), dtype=torch.long)
    X_valid = torch.zeros((100 * samples_per_class_val, 3, 32, 32))
    y_valid = torch.zeros((100 * samples_per_class_val, ), dtype=torch.long)

    for i in range(100):
        index_y = np.where(np.array(cifar100_train.targets) == i)[0]
        np.random.shuffle(index_y)
        X_train[i * eff_samples_class:(i + 1) * eff_samples_class] = X[index_y[:eff_samples_class]]
        y_train[i * eff_samples_class:(i + 1) * eff_samples_class] = torch.tensor(cifar100_train.targets)[index_y[:eff_samples_class]].long()
        X_valid[i * samples_per_class_val:(i + 1) * samples_per_class_val] = X[index_y[eff_samples_class:]]
        y_valid[i * samples_per_class_val:(i + 1) * samples_per_class_val] = torch.tensor(cifar100_train.targets)[index_y[eff_samples_class:]].long()

    # Create Test set
    X_test = X[50000:]
    y_test = torch.tensor(cifar100_test.targets).long()

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test
    )


def iterate_minibatches(
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        batchsize: int, 
        shuffle: bool = False, 
        augment: bool = False,
        seed: int = 42
    ):
    """
    Generate minibatches from inputs and targets.

    Parameters
    ----------
    inputs : torch.Tensor
        Input data.

    targets : torch.Tensor
        Target data.

    batchsize : int
        Batch size.

    shuffle : bool
        Shuffle the data.

    augment : bool
        Augment the data.

    seed : int
        Random seed.
    """
    np.random.seed(seed)
    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if augment:
            # pad feature arrays with 4 pixels on each side of images of 32x32
            # and do random cropping of 32x32
            padded = F.pad(inputs[excerpt], (4, 4, 4, 4), mode='constant', value=0)
            random_cropped = torch.zeros(inputs[excerpt].shape)
            crops = torch.randint(0, 9, size=(batchsize, 2))
            for r in range(batchsize):
                # Cropping and possible flipping
                if (np.random.randint(2) > 0):
                    random_cropped[r] = padded[r, :, crops[r, 0]:crops[r, 0] + 32, crops[r, 1]:crops[r, 1] + 32]
                else:
                    random_cropped[r] = torch.flip(padded[r, :, crops[r, 0]:crops[r, 0] + 32, crops[r, 1]:crops[r, 1] + 32], [2])

            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]
    

      
