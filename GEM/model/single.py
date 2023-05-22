import torch
import torch.nn as nn
from .mlp import MLP

class Net(nn.Module):
    """
    A single predictor trained across all tasks
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_tasks: int,
        args
    ):
        """
        Parameters
        ----------
        n_inputs : int
            Number of inputs

        n_outputs : int
            Number of outputs

        n_tasks : int
            Number of tasks

        args
            Command line arguments
        """
        super(Net, self).__init__()
        num_layers = args.num_layers
        hidden_size = args.hidden_size

        self.net = MLP([n_inputs] + [hidden_size] * num_layers + [n_outputs])

        # Set up optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # Set up losses
        self.loss = nn.CrossEntropyLoss()
        self.num_classes_per_task = n_outputs
        self.n_outputs = n_outputs


    def forward(self, x, t):
        output = self.net(x)
        return output
    
    def observe(self, x, t, y):
        self.train()
        self.opt.zero_grad()
        output = self.forward(x, t)
        loss = self.loss(output, y)
        loss.backward()
        self.opt.step()