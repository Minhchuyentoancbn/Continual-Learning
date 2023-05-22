import torch
import torch.nn as nn
import torch.optim as optim
from .mlp import MLP

import numpy as np
import quadprog


def store_grad(pp, grads, grad_dims, tid: int):
    """
    Store the gradients into a buffer

    Parameters
    ----------
    pp : torch.nn.ParameterList
        List of parameters

    grads : torch.Tensor
        Gradients

    grad_dims: list
        List with number of parameters per layers

    tid : int
        Task ID
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.

    Parameters
    ----------
    pp : torch.nn.ParameterList
        List of parameters

    newgrad : torch.Tensor
        New gradient

    grad_dims: list
        List with number of parameters per layers
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            param.grad.data.copy_(newgrad[beg:en].contiguous().view(param.grad.data.size()))
        cnt += 1


def project2cone2(gradient: torch.Tensor, memories: torch.Tensor, margin:float=0.5, eps:float=1e-3):
    """
    Solves the GEM dual QP described in the paper given a proposed
    gradient "gradient", and a memory of task gradients "memories".
    Overwrites "gradient" with the final projected update.

    Parameters
    ----------
    gradient: torch.Tensor
        p-vector
    
    memories: torch.Tensor
        (t * p)-vector

    margin: float
        Margin add to v* biased the gradient projection to updates that favoured benefitial backwards transfer.

    eps: float
        Epsilon tolerance parameter for the quadprog solver.

    Returns
    -------
    x: torch.Tensor
        p-vector

    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    """
    Gradient Episodic Memory for Continual Learning
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
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.num_classes_task = n_outputs

        # Set up the episodic memory
        self.margin = args.memory_strength
        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # Allocate episodic memory
        self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if self.gpu:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # Allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if self.gpu:
            self.grads = self.grads.cuda()

        # Allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0



    def forward(self, x, t):
        output = self.net(x)
        return output
    
    def observe(self, x, t, y):
        # Update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
        batch_size = y.data.size(0)
        end_count = min(self.mem_cnt + batch_size, self.n_memories)
        eff_batch_size = end_count - self.mem_cnt
        self.memory_data[t, self.mem_cnt:end_count].copy_(x.data[:eff_batch_size])

        if batch_size == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt:end_count].copy_(y.data[:eff_batch_size])

        self.mem_cnt += eff_batch_size
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0  # Reset for next task

        # Compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]
                output = self.forward(self.memory_data[past_task], past_task)
                past_task_loss = self.ce(output, self.memory_labs[past_task])
                past_task_loss.backward()

                store_grad(self.parameters, self.grads, self.grad_dims, past_task)

        # Compute gradient on current task
        self.zero_grad()
        output = self.forward(x, t)
        loss = self.ce(output, y)
        loss.backward()

        # Check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)
                
        self.opt.step()