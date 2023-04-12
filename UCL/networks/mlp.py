import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Iterable


class Net(torch.nn.Module):
    """
    MLP with 2 hidden layers
    """
    
    def __init__(self, input_size: Union[int, int, int],
                 taskcla: Iterable[Union[int, int]],
                 units: int = 400, split: bool = False,
                 notMNIST: bool = False):
        """
        Parameters
        ----------
        input_size : Union[int, int, int]
            Size of each input sample.

        taskcla : Iterable[Union[int, int]]
            Number of classes in each task.

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
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(n_chanels * size * size, units)
        self.fc2 = nn.Linear(units, units)

        if notMNIST:
            self.fc3 = nn.Linear(units, units)
            self.fc4 = nn.Linear(units, units)

        if split:
            self.last = nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(nn.Linear(units, n))
        else:
            self.last = nn.Linear(units, taskcla[0][1])

        
    def forward(self, x: torch.Tensor):
        h = x.view(x.size(0), -1)
        h = self.drop(F.relu(self.fc1(h)))
        h = self.drop(F.relu(self.fc2(h)))
        # h = F.relu(self.fc1(h))
        # h = F.relu(self.fc2(h))
        if self.notMNIST:
            h = self.drop(F.relu(self.fc3(h)))
            h = self.drop(F.relu(self.fc4(h)))
            # h = F.relu(self.fc3(h))
            # h = F.relu(self.fc4(h))

        if self.split:
            y = []
            for t, n in self.taskcla:
                y.append(self.last[t](h))
        else:
            y = self.last(h)

        return y