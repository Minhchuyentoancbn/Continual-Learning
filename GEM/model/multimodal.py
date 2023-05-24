import torch
import torch.nn as nn
import torch.nn.functional as F

def reset_bias(m):
    m.bias.data.fill_(0.0)


class Net(nn.Module):
    """
    Has the same architecture of `single`, but with a dedicated
    input layer per task
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

        self.input_layers = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        self.num_layers = args.num_layers
        hidden_size = args.hidden_size

        if self.num_layers > 0:
            # Dedicated input layers
            for _ in range(n_tasks):
                self.input_layers.append(nn.Linear(n_inputs, hidden_size))
                self.input_layers[-1].apply(reset_bias)

            # Hidden layers
            self.hidden_layers.append(nn.ModuleList())
            for _ in range(self.num_layers):
                self.hidden_layers[0].append(nn.Linear(hidden_size, hidden_size))
                self.hidden_layers[0][0].apply(reset_bias)

            # Shared output layer
            self.output_layers.append(nn.Linear(hidden_size, n_outputs))
            self.output_layers[-1].apply(reset_bias)
        else:
            # No hidden layers
            self.input_layers.append(nn.Linear(n_inputs, hidden_size))
            self.input_layers[-1].apply(reset_bias)

        self.relu = nn.ReLU()
        self.soft = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)


    def forward(self, x, t):
        h = x

        if self.num_layers == 0:
            y = self.soft(self.input_layers[t if isinstance(t, int) else t[0]](h))
        else:
            # task-specific input layers
            h = self.relu(self.input_layers[t if isinstance(t, int) else t[0]](h))
            for l in range(self.num_layers):
                h = self.relu(self.hidden_layers[0][l](h))

            # Shared output layer
            y = self.soft(self.output_layers[0](h))

        return y
    
    def observe(self, x, t, y):
        self.train()
        self.zero_grad()
        output = self.forward(x, t)
        loss = self.loss(output, y)
        loss.backward()
        self.opt.step()
