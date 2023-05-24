import torch
import torch.nn as nn
from .mlp import MLP

class Net(nn.Module):
    """
    One independent predictor per task. Each independent predictor has the same architecture as
    `single` but with T times less hidden units than `single`. Each new independent predictor can
    be initialized at random, or be a clone of the last trained predictor
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

        self.net = nn.ModuleList()
        self.opts = []

        # Set up network
        for _ in range(n_tasks):
            self.net.append(MLP([n_inputs] + [int(hidden_size / n_tasks)] * num_layers + [n_outputs]))
        
        # Set up optimizer
        for t in range(n_tasks):
            self.opts.append(torch.optim.SGD(self.net[t].parameters(), lr=args.lr))

        # Set up losses
        self.loss = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.old_task = 0
        self.finetune = args.finetune
        self.gpu = args.cuda

    def forward(self, x, t):
        output = self.net[t](x)
        return output
    
    def observe(self, x, t, y):
        # detect beginning of a new task
        if self.finetune and t > 0 and t != self.old_task:
            # copy old task to new task
            for old_param, new_param in zip(self.net[self.old_task].parameters(), self.net[t].parameters()):
                new_param.data.copy_(old_param.data)

            self.old_task = t

        self.train()
        self.zero_grad()
        output = self.forward(x, t)
        loss = self.loss(output, y)
        loss.backward()
        self.opts[t].step()