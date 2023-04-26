import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from utils import *
from typing import Union, List

sys.path.append('..')

class Approach:
    """
    Class implementing the Hard Attention to the Task approach
    """

    def __init__(self,
                 model: nn.Module,
                 epochs: int = 100,
                 batch_size: int = 256,
                 lr: float = 0.05,
                 lr_factor: float = 3,
                 lr_patience: int = 5,
                 clip_grad: float = 10000,
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

        self.model = model
        self.epochs = epochs
        self.iteration = 0
        self.batch_size = batch_size
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_min = 1e-4
        self.clip_grad = clip_grad

        self.args = args
        file_name = log_name
        self.logger = logger(file_name, path='./result_data/csvdata/', data_format='csv')

        self.ce = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        self.lamb = args['alpha']
        self.smax = args['smax']

        self.mask_pre = None  # mask for the previous task a(<=t)
        self.mask_back = None  # mask for the backward pass
        self.split = split

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
        task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
        mask = self.model.mask(task, s= self.smax)
        for i in range(len(mask)):
            mask[i] = torch.autograd.Variable(mask[i].data.clone(), requires_grad=False)
        if t == 0:
            self.mask_pre = mask

        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i] = torch.max(self.mask_pre[i], mask[i])

        # Weight mask
        self.mask_back = dict()
        for n, p in self.model.named_parameters():
            vals = self.model.get_view_for(n, self.mask_pre)
            if vals is not None:
                self.mask_back[n] = 1 - vals

        return


    def train_epoch(self, 
                    t: int, 
                    X: torch.Tensor, 
                    y: torch.Tensor,
                    thres_cosh=50,
                    thres_emb=6
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
            images = torch.autograd.Variable(X[r[i:i + self.batch_size]])
            targets = torch.autograd.Variable(y[r[i:i + self.batch_size]])
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())

            s = (self.smax - 1 / self.smax) * i / len(r) + 1 / self.smax
            # Forward pass
            output, masks = self.model(task, images, s)
            if self.split:
                output = output[t]

            loss, _ = self.criterion(output, targets, masks)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Restrict layer gradients in backprop
            if t > 0:
                for n, p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data *= self.mask_back[n]

            # Compensate embedding gradients
            for n, p in self.model.named_parameters():
                if n.startswith('e'):
                    num = torch.cosh(torch.clamp(s*p.data, -thres_cosh, thres_cosh)) + 1
                    den = torch.cosh(p.data) + 1
                    p.grad.data *= self.smax/s*num/den

            # Apply step
            if self.args['optimizer'] == 'sgd' or self.args['optimizer'] == 'sgd_momentum_decay':
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            # Constrain embeddings
            for n, p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data = torch.clamp(p.data, -thres_emb, thres_emb)

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

        total_reg = 0
        r = np.arange(X.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop over the batches
        with torch.no_grad():
            for i in range(0, X.size(0), self.batch_size):
                # Get the inputs
                images = torch.autograd.Variable(X[r[i:i + self.batch_size]])
                targets = torch.autograd.Variable(y[r[i:i + self.batch_size]])
                task = torch.autograd.Variable(torch.LongTensor([t]).cuda())

                output, masks = self.model(task, images, self.smax)
                if self.split:
                    output = output[t]

                loss, reg = self.criterion(output, targets, masks)
                acc = (output.max(1)[1] == targets).sum().item()

                total_loss += loss.item() * len(targets)
                total_acc += acc
                total_num += len(targets)
                total_reg += reg * len(targets)

        print('  {:.3f}  '.format(total_reg/total_num),end='')

        return total_loss / total_num, total_acc / total_num



    def criterion(self, output: torch.Tensor, targets: torch.Tensor, masks: List[torch.Tensor]) -> Union[torch.Tensor, torch.Tensor]:
        """
        The loss function

        Parameters
        ----------
        output : torch.Tensor
            The output of the model

        targets : torch.Tensor
            The targets

        masks : List[torch.Tensor]
            The list of masks

        Returns
        -------
        loss : torch.Tensor
            The loss

        reg : torch.Tensor
            The regularization term
        """

        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):
                aux = 1 - mp
                reg += (m*aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.shape).item()

        reg /= count
        return self.ce(output, targets) + self.lamb * reg, reg