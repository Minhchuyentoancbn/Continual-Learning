import os
import sys
import random
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

from sklearn.feature_extraction import image
from copy import deepcopy
from typing import Iterable, Union, Callable


resnet = models.resnet18(pretrained=True).cuda()
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])


def crop(x: torch.Tensor, patch_size: int, mode: str = 'train') -> torch.Tensor:
    """
    Crop the image to the patch size.

    Parameters
    ----------
    x : torch.Tensor
        Input image.

    patch_size : int
        Patch size.

    mode : str
        Mode of the network. Either 'train' or 'test' or 'valid'.

    Returns
    -------
    output: torch.Tensor
        Cropped image.
    """
    cropped_image = []
    arr_len = len(x)
    if mode == 'train':
        for idx in range(arr_len):
            patch = image.extract_patches_2d(x[idx].data.cpu().numpy(), (patch_size, patch_size), 
                                             max_patches=1)[0]

            # Random horizontal flipping
            if random.random() > 0.5:
                patch = np.fliplr(patch)

            # Random vertical flipping
            if random.random() > 0.5:
                patch = np.flipud(patch)

            # Corupt sorce image
            patch = np.transpose(patch, (2, 0, 1))  # C x H x W
            patch = torch.from_numpy(patch.copy()).float()
            cropped_image.append(patch)

    elif mode == 'valid' or mode == 'test':
        for idx in range(arr_len):
            patch = x[idx].data.cpu().numpy()
            H, W, C = patch.shape

            # Extract the center patch
            patch = patch[H//2 - patch_size//2 : H//2 + patch_size//2, W//2 - patch_size//2 : W//2 + patch_size//2, :]

            # Corupt sorce image
            patch = np.transpose(patch, (2, 0, 1))  # C x H x W
            patch = torch.from_numpy(patch.copy()).float()
            cropped_image.append(patch)

    return torch.stack(cropped_image).view(-1, 3, patch_size, patch_size).cuda()  # N x C x H x W


def print_model_report(model: nn.Module):
    """
    Print the model report.

    Parameters
    ----------
    model : nn.Module
        Model to print the report.
    
    Returns
    -------
    count : int
        Number of parameters in the model.
    """
    print('-' * 100)
    print(model)
    print('Dimensions =',end=' ')
    count = 0
    for p in model.parameters():
        print(p.size(), end=' ')
        count += np.prod(p.size())
    print()
    print(f'Num parameters = {human_format(count)}')
    print('-' * 100)
    return count


def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0

    return f'{num:.1f}{" KMGTP"[magnitude]}'


def print_optimizer_config(optimizer: optim.Optimizer):
    """
    Print the optimizer configuration.

    Parameters
    ----------
    optimizer : optim.Optimizer
        Optimizer to print the configuration.
    
    Returns
    -------
    None
    """
    if optimizer is None:
        print(optimizer)
    else:
        print(optimizer,'=',end=' ')
        opt = optimizer.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n + ':', opt[n], end=', ')
        print()

    return


def get_model(model: nn.Module) -> dict:
    """
    Get the model state dictionary.

    Parameters
    ----------
    model : nn.Module
        Model to get the state dictionary.

    Returns
    -------
    state_dict : dict
        State dictionary of the model.
    """
    return deepcopy(model.state_dict())


def set_model(model: nn.Module, state_dict: dict):
    """
    Set the model state dictionary.

    Parameters
    ----------
    model : nn.Module
        Model to set the state dictionary.

    state_dict : dict
        State dictionary to set.
    """
    model.load_state_dict(deepcopy(state_dict))
    return


def freeze_model(model: nn.Module):
    """
    Freeze weights of the model.

    Parameters
    ----------
    model : nn.Module
        Model to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False
    return


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1) -> int:
    """
    Compute the output size of the convolution layer.

    Parameters
    ----------
    Lin : int
        Input size.

    kernel_size : int
        Kernel size.

    stride : int
        Stride.

    padding : int
        Padding.

    dilation : int
        Dilation.

    Returns
    -------
    Lout : int
        Output size.
    """
    Lout = int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
    return Lout


def compute_mean_std_dataset(dataset: torch.utils.data.Dataset) -> Union[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and standard deviation of the dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to compute the mean and standard deviation.

    Returns
    -------
    mean : torch.Tensor
        Mean of the dataset.

    std : torch.Tensor
        Standard deviation of the dataset.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    mean = 0.
    std = 0.

    for image, _ in loader:
        mean += image.mean(3).mean(2)  # N x C
    mean /= len(dataset)
    mean_expanded = mean.unsqueeze(2).unsqueeze(3).expand_as(image)  # N x C x H x W

    for image, _ in loader:
        std += ((image - mean_expanded)**2).sum(3).sum(2)  # N x C
    std = (std / (len(dataset) * image.size(2) * image.size(3) - 1)).sqrt()

    return mean, std


def fisher_matrix_diag(t: int, x: torch.Tensor, y: torch.Tensor, 
                       model: nn.Module, criterion: Callable, 
                       sbatch: int = 20, split: bool = False,
                       args: dict = None) -> dict:
    """
    Compute the diagonal of the Fisher information matrix.

    Parameters
    ----------
    t : int
        Task index.

    x : torch.Tensor
        Input tensor.

    y : torch.Tensor
        Label tensor.

    model : nn.Module
        Model to compute the diagonal of the Fisher information matrix.

    criterion : Callable
        Loss function.

    sbatch : int
        Batch size.

    split : bool
        Split the model.

    args : dict
        Arguments.

    Returns
    -------
    fisher : dict
        Diagonal of the Fisher information matrix.
    """

    # Init
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p.data)

    # Compute the diagonal of the Fisher information matrix
    model.train()

    for i in range(0, x.size(0), sbatch):
        # Forward pass
        images = x[i:i+sbatch]
        targets = y[i:i+sbatch]

        size = images.size(0)

        if args['experiment'] == 'split_CUB200':
            images = feature_extractor(images)

        if split:
            output = model(images)[t]
        else:
            output = model(images)

        loss = criterion(t, output, targets)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Compute the diagonal of the Fisher information matrix
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data**2 * size

    # Mean
    with torch.no_grad():
        for n, p in model.named_parameters():
            fisher[n] /= x.size(0)

    return fisher


def cross_entropy(outputs: torch.Tensor, targets: torch.Tensor, exp: int = 1, size_average: bool = True, eps: float = 1e-5) -> torch.Tensor:
    """
    Cross entropy.

    Parameters
    ----------
    outputs : torch.Tensor
        Output logits tensor.

    targets : torch.Tensor
        Target logits tensor.

    exp : int
        Exponent to raise the softmax.

    size_average : bool
        Size average.

    eps : float
        Epsilon.

    Returns
    -------
    loss : torch.Tensor
        Cross entropy.
    """
    out = F.softmax(outputs)
    tar = F.softmax(targets)

    if exp != 1:
        # Raise to the power
        out = out**exp
        tar = tar**exp
        # Normalize
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)

    out = out + eps / out.size(1)
    # Normalize
    out = out / out.sum(1).view(-1, 1).expand_as(out)

    # Cross entropy
    ce = - (tar * torch.log(out)).sum(1)
    if size_average:
        ce = ce.mean()
    
    return ce


def set_req_grad(layer: nn.Module, req_grad: bool):
    """
    Set the requires_grad attribute of the layer.

    Parameters
    ----------
    layer : nn.Module
        Layer to set the requires_grad attribute.

    req_grad : bool
        Value of the requires_grad attribute.
    """
    for param in layer.parameters():
        param.requires_grad = req_grad
    return


def clip_relavance_norm(parameters: Iterable[torch.Tensor] | torch.Tensor, 
                        max_norm: float | int, norm_type: float | int = 2) -> float:
    """
    Clip the gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Parameters
    ----------
    parameters : Iterable[torch.Tensor] | torch.Tensor
        Iterable of parameters to clip.

    max_norm : float | int
        Max norm of the gradients.

    norm_type : float | int (default=2)
        Type of the used p-norm. Can be ``'inf'`` for infinity norm.
    
    Returns
    -------
    total_norm : float
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = list(filter(lambda p: p is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Compute the total norm
    if norm_type == float('inf'):
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)

    return total_norm


class logger(object):
    def __init__(self, file_name='pmnist2', resume=False, path='./result_data/csvdata/', data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format


    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = pd.concat([self.log, df], axis=0, ignore_index=True)


    def save(self):
        return self.log.to_csv(self.data_path, index=False, index_label=False)


    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))