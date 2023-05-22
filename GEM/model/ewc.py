import torch
import torch.nn as nn
from .mlp import MLP

class Net(nn.Module):
    """
    EWC [Kirkpatrick et al., 2017], where the loss is regularized to avoid catastrophic forgetting
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
        self.reg = args.memory_strength

        self.net = MLP([n_inputs] + [hidden_size] * num_layers + [n_outputs])

        # Set up optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # Set up losses
        self.loss = nn.CrossEntropyLoss()
        self.num_classes_per_task = n_outputs
        self.n_outputs = n_outputs
        self.n_memories = args.n_memories

        # Set up memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.memx = None
        self.memy = None


    def forward(self, x, t):
        output = self.net(x)
        return output
    
    def observe(self, x, t, y):
        self.train()

        if t != self.current_task:
            self.opt.zero_grad()

            # Compute Fisher
            self.loss(self(self.memx, self.current_task), self.memy).backward()
            self.fisher[self.current_task] = []
            self.optpar[self.current_task] = []

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.fisher[self.current_task].append(pg)
                self.optpar[self.current_task].append(pd)

            self.current_task = t
            self.memx = None
            self.memy = None

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            if self.memx.size()[0] < self.n_memories:
                self.memx = torch.cat((self.memx, x.data.clone()), 0)
                self.memy = torch.cat((self.memy, y.data.clone()), 0)
                if self.memx.size()[0] > self.n_memories:
                    self.memx = self.memx[:self.n_memories]
                    self.memy = self.memy[:self.n_memories]

        self.opt.zero_grad()
        loss = self.loss(self(x, t), y)
        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()

        loss.backward()
        self.opt.step()