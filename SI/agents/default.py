import torch
import torch.nn as nn
import torch.optim as optim
import models

from types import MethodType
from utils.metrics import accuracy, AverageMeter, Timer
from typing import Dict


class NormalNN(nn.Module):
    """
    Normal Neural Network for classification.
    """
    
    def __init__(self, agent_config: dict) -> None:
        """
        Parameters
        ----------
        agent_config : dict
            A dictionary of configuration for the agent.
            Keys:
                - epochs: int, the number of epochs.
                - lr: float, the learning rate.
                - weight_decay: float, the weight decay.
                - reg_coef: float, the regularization coefficient.
                - model_type: str, the type of the model.
                - model_name: str, the name of the model.
                - out_dim: dict, {task:dim}.
                - model_weights: str, the path to the model weights.
                - print_freq: int, the frequency of printing.
                - gpu: bool, whether to use gpu.

        Returns
        -------
        None.
        """
        
        super(NormalNN, self).__init__()
        self.config = agent_config

        # Whether to log the training process
        if agent_config['print_freq'] > 0:
            self.log = True
        else:
            self.log = False

        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(self.config['out_dim']) > 1 else False
        self.model = self.create_model()
        self.criterion_fn = nn.CrossEntropyLoss()

        if self.config['gpu']:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False

        self.init_optimizer()
        self.reset_optimizer = False

        self.valid_out_dim = 'All'  # Default: 'All' means all ouput nodes are active
                                    # Set a interger here for the incremental class scenario



    def create_model(self) -> nn.Module:
        """
        Create the model based on the configuration.

        Returns
        -------
        model : nn.Module
            The created model.
        """

        # Define the backbone model
        model = models.__dict__[self.config['model_type']].__dict__[self.config['model_name']]()

        # Create heads fir tasks (it can be single tasl or multi-task)
        n_feat = model.last.in_features
        model.last = nn.ModuleDict()
        for task, out_dim in self.config['out_dim'].items():
            model.last[task] = nn.Linear(n_feat, out_dim)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs
        
        # Replace the logits function
        model.logits = MethodType(new_logits, model)

        # Load the pre-trained weights
        if self.config['model_weights'] is not None:
            print('=> Loading pre-trained weights from {}'.format(self.config['model_weights']))
            model_state = torch.load(
                self.config['model_weights'],
                map_location=lambda storage, loc: storage # Load on CPU
            )
            model.load_state_dict(model_state)
            print('=> Load Done')

        return model


    def init_optimizer(self):
        """
        Initialize optimizer.

        Returns
        -------
        None
        """
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)
    

    def predict(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict the output given the input.

        Parameters
        ----------
        inputs: torch.Tensor
            The input tensor.

        Returns
        -------
        output : Dict[str, torch.Tensor]
            The output of the model.
        """
        self.model.eval()
        out =  self.model(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return out
    

    def validation(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Validation.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            dataloader

        Returns
        -------
        acc : float
            accuracy
        """
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    input, target = input.cuda(), target.cuda()
        
            output = self.predict(input)
            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, target, task, acc)

        self.train(orig_mode)

        if self.log:
            print('Validation: Time: {:.3f} Acc: {:.3f}'.format(batch_timer.toc(), acc.avg))
        
        return acc.avg


    def criterion(self, preds: Dict[str, torch.Tensor], targets: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss.

        Parameters
        ----------
        preds : Dict[str, torch.Tensor]
            The prediction tensor.

        targets : torch.Tensor
            The target tensor.

        tasks : torch.Tensor
            The task tensor.

        Returns
        -------
        loss : torch.Tensor
            The loss.
        """
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.

        if self.multihead:
            loss = 0
            for t, t_preds in preds.items():
                indices = [i for i in range(len(tasks)) if tasks[i] == t]
                if len (indices) > 0:
                    t_preds = t_preds[indices]
                    t_targets = targets[indices]
                    loss += self.criterion_fn(t_preds, t_targets) * len(indices)  # Restore the loss from average
            
            loss /= len(targets)
        else:
            pred = preds['All']
            # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
            if isinstance(self.valid_out_dim, int):
                pred = preds['All'][:, :self.valid_out_dim]
            loss = self.criterion_func(pred, targets)
        
        return loss
    

    def update_model(self, inputs: torch.Tensor, targets: torch.Tensor, tasks: torch.Tensor) -> float:
        """
        Update the model.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor.

        targets : torch.Tensor
            The target tensor.

        tasks : torch.Tensor
            The task tensor.

        Returns
        -------
        loss : torch.Tensor
            The loss.
        output : Dict[str, torch.Tensor]
            The output of the model.
        """
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out
    

    def learn_batch(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader = None) -> float:
        """
        Learn a batch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The training dataloader.

        val_loader : torch.utils.data.DataLoader
            The validation dataloader.

        Returns
        -------
        None
        """
        # Reset optimizer before learning each task
        if self.reset_optimizer:
            if self.log:
                print('Reset optimizer')
            self.init_optimizer()

        for epoch in range(self.config['epochs']):
            batch_timer = Timer()
            data_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and the optimizer
            if self.log:
                print('Epoch: {}'.format(epoch))
            self.model.train()

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            if self.log:
                print('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            
            for i , (inputs, targets, tasks) in enumerate(train_loader):
                # measure data loading time
                data_time.update(data_timer.toc())

                if self.gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                loss_batch, output = self.update_model(inputs, targets, tasks)
                targets = targets.detach()
                inputs = inputs.detach()

                # measure accuracy and record loss
                acc = accumulate_acc(output, targets, tasks, acc)
                losses.update(loss_batch, inputs.size(0))

                batch_time.update(batch_timer.toc())
                data_timer.toc()

                if self.log and i % self.config['print_freq'] == 0:
                    print('[{0}/{1}]\t{batch_time.val:.4f} ({batch_time.avg:.4f})\t{data_time.val:.4f} ({data_time.avg:.4f})\t{loss.val:.3f} ({loss.avg:.3f})\t{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))
                    
            if self.log:
                print(f'Train Acc: {acc.avg:.3f}')

            if val_loader is not None:
                self.validation(val_loader)
    

    def add_valid_output_dim(self, dim: int = 0):
        """
        Add the valid output dimension to support incremental learning.

        Parameters
        ----------
        dim : int (default=0)
            The valid output dimension.

        Returns
        -------
        valid_out_dim : int
            The valid output dimension.
        """
        if self.log:
            print('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'All':
            self.valid_out_dim = 0
        
        self.valid_out_dim += dim
        if self.log:
            print('Incremental class: New valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim


    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())


    def cuda(self):
        """
        Move model to cuda.
        """
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        return self


    def save_model(self, filename: str) -> None:
        """
        Save model.

        Parameters
        ----------
        filename : str
            The filename.

        Returns
        -------
        None
        """
        model_state = self.model.state_dict()
        for key in model_state.keys():
            model_state[key] = model_state[key].cpu()
            
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')


def accumulate_acc(output: Dict[str, torch.Tensor], target: torch.Tensor, task: torch.tensor, meter: AverageMeter) -> AverageMeter:
    """
    Accumulate the accuracy.

    Parameters
    ----------
    output : Dict[str, torch.Tensor]
        The output of the model.

    target : torch.Tensor
        The target tensor.

    task : torch.tensor
        The task tensor.

    meter : AverageMeter
        The accuracy meter.

    Returns
    -------
    meter : AverageMeter
        The updated accuracy meter.
    """
    if 'All' in output.keys(): # Single-headed model
        meter.update(accuracy(output['All'], target), len(target))

    else:  # outputs from multi-headed (multi-task) model
        for t, t_out in output.items():
            inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
            if len(inds) > 0:
                t_out = t_out[inds]
                t_target = target[inds]
                meter.update(accuracy(t_out, t_target), len(inds))

    return meter
