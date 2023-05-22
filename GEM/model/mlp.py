import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-layer perceptron
    """
    def __init__(self, sizes: list):
        """
        Parameters
        ----------
        sizes : list[int]
            List of layer sizes
        """
        super(MLP, self).__init__()
        layers = []

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                self.layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)