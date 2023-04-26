import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Iterable, List

class Net(nn.Module):
    """
    HAT MLP with 2 hidden layers
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
        if notMNIST:
            units = 150

        self.notMNIST = notMNIST
        self.taskcla = taskcla
        self.split = split
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(n_chanels * size * size, units)
        self.embedding_fc1 = nn.Embedding(len(self.taskcla), units)
        self.fc2 = nn.Linear(units, units)
        self.embedding_fc2 = nn.Embedding(len(self.taskcla), units)

        if notMNIST:
            self.fc3 = nn.Linear(units, units)
            self.embedding_fc3 = nn.Embedding(len(self.taskcla), units)
            self.fc4 = nn.Linear(units, units)
            self.embedding_fc4 = nn.Embedding(len(self.taskcla), units)

        if split:
            self.last = nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(nn.Linear(units, n))

        else:
            self.last = nn.Linear(units, taskcla[0][1])


        self.gate = nn.Sigmoid()


    def forward(
            self,
            t: int,
            x: torch.Tensor,
            s: float = 1
    ):
        """
        Forward pass of the network

        Parameters
        ----------
        t : int
            Task index.

        x : torch.Tensor
            Input tensor.

        s : float, optional
            Scaling factor. The default is 1.

        Returns
        -------
        y: torch.Tensor
            Output tensor.

        masks: List[torch.Tensor]
            List of masks.
        """
        # Gates
        gate_fc1 = self.gate(s*self.embedding_fc1(t))
        gate_fc2 = self.gate(s*self.embedding_fc2(t))

        # Gated
        h = self.drop2(x.view(x.size(0), -1))
        h = self.drop1(F.relu(self.fc1(h)))
        h = h * gate_fc1.expand_as(h)
        h = self.drop1(F.relu(self.fc2(h)))
        h = h * gate_fc2.expand_as(h)

        if self.notMNIST:
            gate_fc3 = self.gate(s*self.embedding_fc3(t))
            gate_fc4 = self.gate(s*self.embedding_fc4(t))
            h = self.drop1(F.relu(self.fc3(h)))
            h = h * gate_fc3.expand_as(h)
            h = self.drop1(F.relu(self.fc4(h)))
            h = h * gate_fc4.expand_as(h)

        if self.split:
            y = []
            for t, _ in self.taskcla:
                y.append(self.last[t](h))
        else:
            y = self.last(h)


        masks = [gate_fc1, gate_fc2]
        if self.notMNIST:
            masks += [gate_fc3, gate_fc4]

        return y, masks
    

    def mask(self, t: int, s: float = 1) -> List[torch.Tensor]:
        """
        Get the masks for the current task

        Parameters
        ----------
        t : int
            Task index.

        s : float, optional
            Scaling factor. The default is 1.

        Returns
        -------
        List[torch.Tensor]
            List of masks.
        """
        gate_fc1 = self.gate(s*self.embedding_fc1(t))
        gate_fc2 = self.gate(s*self.embedding_fc2(t))

        if self.notMNIST:
            gate_fc3 = self.gate(s*self.embedding_fc3(t))
            gate_fc4 = self.gate(s*self.embedding_fc4(t))
            return [gate_fc1, gate_fc2, gate_fc3, gate_fc4]
        
        return [gate_fc1, gate_fc2]

        
    def get_view_for(self, n: str, masks: List[torch.Tensor]):
        """
        Compute min(a(l, i), a(l+1, j))

        Parameters
        ----------
        n : str
            Name of the layer.

        masks : List[torch.Tensor]
            List of masks.

        Returns
        -------
        torch.Tensor
        """
        if self.notMNIST:
            gate_fc1, gate_fc2, gate_fc3, gate_fc4 = masks
        else:
            gate_fc1, gate_fc2 = masks

        if n == 'fc1.weight':
            return gate_fc1.data.view(-1, 1).expand_as(self.fc1.weight)
        elif n == 'fc1.bias':
            return gate_fc1.data.view(-1)
        
        if n == 'fc2.weight':
            post = gate_fc2.data.view(-1, 1).expand_as(self.fc2.weight)
            pre = gate_fc1.data.view(1, -1).expand_as(self.fc2.weight)
            return torch.min(post, pre)
        elif n == 'fc2.bias':
            return gate_fc2.data.view(-1)
        
        if self.notMNIST:
            if n == 'fc3.weight':
                post = gate_fc3.data.view(-1, 1).expand_as(self.fc3.weight)
                pre = gate_fc2.data.view(1, -1).expand_as(self.fc3.weight)
                return torch.min(post, pre)
            elif n == 'fc3.bias':
                return gate_fc3.data.view(-1)
            
            if n == 'fc4.weight':
                post = gate_fc4.data.view(-1, 1).expand_as(self.fc4.weight)
                pre = gate_fc3.data.view(1, -1).expand_as(self.fc4.weight)
                return torch.min(post, pre)
            elif n == 'fc4.bias':
                return gate_fc4.data.view(-1)
            
        return None
        