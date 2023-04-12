import torch
import torch.nn as nn
import torch.nn.functional as F

from bayes_layer import BayesianLinear
from typing import Union, Iterable


class Net(torch.nn.Module):
    """
    UCL MLP with 2 hidden layers
    """
    
    def __init__(self, input_size: Union[int, int, int],
                 taskcla: Iterable[Union[int, int]],
                 ratio: float,
                 units: int = 400, split: bool = False,
                 notMNIST: bool = False):
        """
        Parameters
        ----------
        input_size : Union[int, int, int]
            Size of each input sample.

        taskcla : Iterable[Union[int, int]]
            Number of classes in each task.

        ratio : float
            Ratio of init standard deviation (see UCL paper)

        units : int, optional
            Number of units in hidden layers. The default is 400.

        split : bool, optional
            Whether to split by task. The default is False.

        notMNIST : bool, optional
            Whether to use notMNIST dataset. The default is False.
        """
        super(Net, self).__init__()

        n_chanels, size, _ = input_size
        self.notMNIST = notMNIST
        if notMNIST:
            units = 150
        self.taskcla = taskcla
        self.split = split
        self.fc1 = BayesianLinear(n_chanels * size * size, units, ratio)
        self.fc2 = BayesianLinear(units, units, ratio)

        if notMNIST:
            self.fc3 = BayesianLinear(units, units, ratio)
            self.fc4 = BayesianLinear(units, units, ratio)

        if split:
            self.last = nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(nn.Linear(units, n))
        else:
            self.last = BayesianLinear(units, taskcla[0][1], ratio)

        
    def forward(self, x: torch.Tensor, sample: bool = False):
        h = x.view(x.size(0), -1)
        h = F.relu(self.fc1(h, sample))
        h = F.relu(self.fc2(h, sample))
        if self.notMNIST:
            h = F.relu(self.fc3(h, sample))
            h = F.relu(self.fc4(h, sample))

        if self.split:
            y = []
            for t, n in self.taskcla:
                y.append(self.last[t](h))
        else:
            y = self.last(h, sample)
            y = F.log_softmax(y, dim=1)

        return y