import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _single, _pair, _triple
from typing import Union


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> Union[int, int]:
    """Computes the fan in and fan out of a tensor.

    - fan_in is the number of inputs to a layer (4)
    - fan_out is the number of outputs to a layer (6)

    Example:
    >>> from torch import nn
    >>> conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=2)
    >>> print(conv.weight.shape)
        torch.Size([1, 1, 2, 2])
    >>> print(nn.init._calculate_fan_in_and_fan_out(conv.weight))
        (4, 4)

    See the definitin here: 
    https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:  # Convolutional layers
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)

        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()

        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class Gaussian(object):
    """
    Random parameters that have a Gaussian distribution.
    """
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        """
        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution.

        rho : torch.Tensor
            log(exp(std) - 1) where std is the standard deviation of the distribution.
        """
        super().__init__()
        self.mu = mu.cuda()
        self.rho = rho.cuda()
        self.normal = torch.distributions.Normal(0, 1)


    @property
    def sigma(self) -> torch.Tensor:
        # Save rho instead of sigma to ensure positivity
        return torch.log1p(torch.exp(self.rho))
    

    def sample(self):
        """
        Sample values of the parameters
        """
        epsilon = self.normal.sample(self.mu.size()).cuda()
        return self.mu + self.sigma * epsilon


class BayesianLinear(nn.Module):
    """
    Linear layer with Bayesian weights.
    """

    def __init__(self, in_features: int, out_features: int, ratio: float=0.5):
        """
        Parameters
        ----------
        in_features : int
            Number of input features.

        out_features : int
            Number of output features.

        ratio : float
            Ratio of the noise variance to the total variance.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize the parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features).zero_())
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)

        # var[mu] + sigma^2 = 2 / fan_in = var[w]
        total_var = 2 / fan_in
        noise_var = total_var * ratio  # sigma_init^2
        mu_var = total_var - noise_var  # var[mu]

        noise_std = math.sqrt(noise_var)
        mu_std = math.sqrt(mu_var)
        bound = math.sqrt(3) * mu_std
        rho_init = np.log(np.exp(noise_std) - 1)  # sigma_init rho

        # Initialize the parameters
        nn.init.uniform_(self.weight_mu, -bound, bound)  # var(mu) = 4 * bound^2 / 12 = mu_var

        self.weight_rho = nn.Parameter(torch.Tensor(out_features, 1).fill_(rho_init))

        self.weight = Gaussian(self.weight_mu, self.weight_rho)


    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        
        sample : bool
            Whether to sample the weights or use the mean.

        Returns
        -------
        torch.Tensor
        """
        # Sample the weights
        if sample:
            weight = self.weight.sample()
        else:
            weight = self.weight.mu
            
        bias = self.bias

        return F.linear(x, weight, bias)
    

class _BayesianConvNd(nn.Module):
    """
    Parent class for Bayesian convolutional layers.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride, padding, dilation, transposed, output_padding,
                 groups: int = 1, bias: bool = True, ratio: float=0.5):
        
        super(_BayesianConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        # Initialize the parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels).zero_(), requires_grad=bias)

        _, fan_out = _calculate_fan_in_and_fan_out(self.weight_mu)

        total_var = 2 / fan_out
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std = math.sqrt(noise_var)
        mu_std = math.sqrt(mu_var)
        bound = math.sqrt(3) * mu_std
        rho_init = np.log(np.exp(noise_std) - 1)  # sigma_init rho

        
        nn.init.uniform_(self.weight_mu, -bound, bound)
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1).fill_(rho_init))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)


class BayesianConv2d(_BayesianConvNd):
    """
    Bayesian convolutional layer.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups: int = 1, bias: bool = True, ratio: float=0.25):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        kernel_size : int or tuple
            Size of the convolutional kernel.

        stride : int or tuple
            Stride of the convolution.

        padding : int or tuple
            Padding added to both sides of the input.

        dilation : int or tuple
            Spacing between kernel elements.

        groups : int
            Number of blocked connections from input channels to output channels.

        bias : bool
            Whether to add a bias term to the output.

        ratio : float
            Ratio of the noise variance to the total variance.

        """
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesianConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, ratio
        )

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        
        sample : bool
            Whether to sample the weights or use the mean.

        Returns
        -------
        torch.Tensor
        """
        if sample:
            weight = self.weight.sample()
        else:
            weight = self.weight.mu

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)