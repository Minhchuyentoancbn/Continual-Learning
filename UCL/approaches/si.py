import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from utils import *
from typing import Union

sys.path.append('..')


class Approach:
    """
    Class implementing the Synaptic intelligence approach
    """

    def __init__(self,
                 model: nn.Module,
                 epochs: int = 100,
                 batch_size: int = 256,
                 lr: float = 0.001,
                 lr_factor: float = 3,
                 lr_patience: int = 5,
                 clip_grad: float = 100,
                 args: dict = None,
                 log_name: str = None,
                 split: bool = False
                ) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The model to train

        epochs : int
            The number of epochs to train

        batch_size : int
            The batch size to use

        lr : float
            The learning rate

        lr_factor : float
            The factor to reduce the learning rate by

        lr_patience : int
            The number of epochs to wait before reducing the learning rate

        clip_grad : float
            The gradient clipping value

        args : dict
            The arguments to use

        log_name : str
            The name of the log file

        split : bool
            Whether to split the data into training and validation sets

        Returns
        -------
        None
        """

        # Set the model
        self.model = model
        self.model_old = deepcopy(model)

        # Set the logger
        file_name = log_name
        self.logger = logger(file_name, path='./result_data/csvdata/', data_format='csv')

        # Set the training parameters
        self.iteration = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_min = 1e-6
        self.lr_patience = lr_patience
        self.clip_grad = clip_grad
        self.split = split

        self.ce = nn.CrossEntropyLoss()
        self.args = args
        self.optimizer = self._get_optimizer()

        # Hyperparameters
        self.c = args['c']
        self.epsilon = 0.01
        if args['experiment'] == 'split_notmnist':
            self.epsilon = 0.001
        self.omega = dict()
        self.W = dict()
        self.p_old = dict()

        # Register starting param-values (needed for “intelligent synapses”).
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
        return


    def _get_optimizer(self, lr=None) -> optim.Optimizer:
        """
        Get the optimizer based on args['optimizer']
        """
        if lr is None:
            lr = self.lr

        if self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr)

        return optimizer
    

    def train(self, 
              t: int, 
              X_train: torch.Tensor, 
              y_train: torch.Tensor, 
              X_valid: torch.Tensor, 
              y_valid: torch.Tensor, 
              data: dict
        ):
        """
        Train the model

        Parameters
        ----------
        t: int

        X_train : torch.Tensor
            The training data

        y_train : torch.Tensor
            The training labels

        X_valid : torch.Tensor
            The validation data

        y_valid : torch.Tensor
            The validation labels

        data: dict
            The data for all tasks
        """
        best_loss = np.inf
        best_model = get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        self.W = dict()
        self.p_old = dict()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.W[n] = p.data.clone().zero_()
                self.p_old[n] = p.data.clone()

        # Trainig loop
        for epoch in range(self.epochs):
            stop = False

            # Train
            clock0 = time.time()
            num_batch = X_train.size(0)
            
            self.train_epoch(t, X_train, y_train)
            clock1 = time.time()
            train_loss, train_acc = self.eval(t, X_train, y_train)
            clock2 = time.time()

            print(
                f'| Epoch {epoch + 1:3d}, time={1000 * self.batch_size * (clock1 - clock0) / num_batch: 5.1f}ms/{1000 * self.batch_size * (clock2 - clock1) / num_batch: 5.1f}ms | '
                f'| Train: loss={train_loss:.3f}, acc={train_acc:5.1f} ', end=''
            )

            # Validation
            valid_loss, valid_acc = self.eval(t, X_valid, y_valid)
            print(f'| Valid: loss={valid_loss:.3f}, acc={valid_acc:5.1f} |', end='')

            # save log for current task & old tasks at every epoch
            self.logger.add(epoch=(t * self.epochs) + epoch, task_num=t + 1, valid_loss=valid_loss, valid_acc=valid_acc)
            for task in range(t):
                X_valid_t = data[task]['valid']['X'].cuda()
                y_valid_t = data[task]['valid']['y'].cuda()
                    
                valid_loss_t, valid_acc_t = self.eval(task, X_valid_t, y_valid_t)
                self.logger.add(epoch=(t * self.epochs) + epoch, task_num=task + 1, valid_loss=valid_loss_t,
                                valid_acc=valid_acc_t)

            # Learning rate scheduler
            if valid_loss < best_loss:
                # Save the best model
                best_loss = valid_loss
                best_model = get_model(self.model)

                # Reset the patience
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1

                # Decay the learning rate
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')

                    if lr < self.lr_min:
                        print()
                        print(f'Stopping training at epoch {epoch + 1}')
                        stop = True

                    # Reset the patience
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

            if stop:
                break
            
        self.logger.save()

        # Restore best
        set_model(self.model, best_model)
        self.update_omega(self.W, self.epsilon)
        self.model_old = deepcopy(self.model)
        freeze_model(self.model_old)  # Freeze the weights

        return


    def train_epoch(self, 
                    t: int, 
                    X: torch.Tensor, 
                    y: torch.Tensor
                    ):
        """
        Train the model for one epoch

        Parameters
        ----------
        t: int
            The current task

        X : torch.Tensor
            The training data

        y : torch.Tensor
            The training labels
        """
        self.model.train()

        r = np.arange(X.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # Loop over the batches
        for i in range(0, X.size(0), self.batch_size):
            self.iteration = self.iteration + 1

            # Get the inputs
            images = X[r[i:i + self.batch_size]]
            targets = y[r[i:i + self.batch_size]]

            # Forward pass
            if self.split:
                output = self.model(images)[t]
            else:
                output = self.model(images)

            loss = self.criterion(t, output, targets)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()

            if self.args['optimizer'] == 'sgd' or self.args['optimizer'] == 'sgd_momentum_decay':
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.p_old[n] = p.detach().clone()

        return


    def eval(self, t: int, X: torch.Tensor, y: torch.Tensor) -> Union[float, float]:
        """
        Evaluate the model on the given data

        Parameters
        ----------
        t: int
            Task index

        X: torch.Tensor
            The input data

        y: torch.Tensor
            The target data

        Returns
        -------
        total_loss: float
            The average loss
        
        total_acc: float
            The average accuracy
        """
        total_loss = 0
        total_acc = 0
        total_num = 0

        self.model.eval()

        # Loop over the batches
        with torch.no_grad():
            for i in range(0, X.size(0), self.batch_size):
                # Get the inputs
                images = X[i:i + self.batch_size]
                targets = y[i:i + self.batch_size]

                # Forward pass
                if self.split:
                    output = self.model(images)[t]
                else:
                    output = self.model(images)

                loss = self.criterion(t, output, targets)
                acc = (output.max(1)[1] == targets).sum().item()

                total_loss += loss.item()
                total_acc += acc
                total_num += len(targets)

        return total_loss / total_num, total_acc / total_num



    def criterion(self, t: int, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        The loss function

        Parameters
        ----------
        t : int
            The current task

        output : torch.Tensor
            The output of the model

        targets : torch.Tensor
            The targets

        Returns
        -------
        loss : torch.Tensor
            The loss
        """

        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            try:
                losses = []
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                        n = n.replace('.', '__')
                        prev_values = getattr(self.model, f'{n}_SI_prev_task')
                        omega = getattr(self.model, f'{n}_SI_omega')
                        losses.append((omega * (p - prev_values) ** 2).sum())

                loss_reg = sum(losses)
            except AttributeError:
                # SI-loss is 0 if there is no stored omega yet
                loss_reg = 0
        
        loss = self.ce(output, targets) + self.c * loss_reg
        return loss
    

    def update_omega(self, W: dict, epsilon: float):
        """
        After completing training on a task, update the per-parameter regularization strength.

        Parameters
        ----------
        W: dict
            Estimated parameter-specific contribution to changes in total loss of completed task
        
        epsilon: float
            Dampening parameter (to bound [omega] when [p_change] goes to 0)
        """
        # Loop over all parameters
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n] / (p_change ** 2 + epsilon)
                try:
                    omega = getattr(self.model, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.model.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.model.register_buffer('{}_SI_omega'.format(n), omega_new)