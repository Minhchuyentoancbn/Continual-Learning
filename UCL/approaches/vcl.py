import sys
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from bayes_layer import BayesianLinear, BayesianConv2d, _calculate_fan_in_and_fan_out
from utils import *
from typing import Union

sys.path.append('..')


class Approach:
    """
    VCL approach
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
        self.param_name = [n for n, p in self.model.named_parameters()]

        # Set the logger
        file_name = log_name
        self.logger = logger(file_name, path='./result_data/csvdata/', data_format='csv')
        self.saved = 0  # Whether the model has been saved

        # Set the training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_grad = clip_grad
        self.epoch = 0  # number of epochs the model has been trained for
        self.iteration = 0  # number of iterations the model has been trained for
        self.args = args
        self.split = split
        self.num_sample = args['num_sample']  # number of time to sample weights
        # self.drop = [20, 40, 60, 75, 90]

        # Set the learning rate
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_min = lr / (lr_factor ** 5)

        # Set the optimizer
        self.optimizer = self._get_optimizer()

        if len(args['parameter']) >= 1:
            params = args['parameter'].split(',')
            print('Setting parameters to', params)
            self.num_sample = int(params[0])

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
            The current task

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

            self.epoch = self.epoch + 1
            
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

            freeze_model(self.model_old)  # Freeze the weights

            if stop:
                break

        # Restore best
        set_model(self.model, best_model)
        self.model_old = deepcopy(self.model)
        self.saved = 1

        self.logger.save()

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
            minibatch_size = len(targets)

            # Sample weights to calulate the negative log-likelihood loss
            sum_sample_loss = 0

            for _ in range(self.num_sample):
                if self.split:
                    output = F.log_softmax(self.model.forward(images, sample=True)[t], dim=1)
                else:
                    output = self.model.forward(images, sample=True)

                sample_loss = F.nll_loss(output, targets, reduction='sum')
                sum_sample_loss += sample_loss

            loss = sum_sample_loss / self.num_sample
            loss = self.custom_regularization(self.model_old, self.model, minibatch_size, loss)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()

            if self.args['optimizer'] == 'sgd' or self.args['optimizer'] == 'sgd_momentum_decay':
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

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
                    output = F.log_softmax(self.model.forward(images, sample=False)[t], dim=1)
                else:
                    output = self.model.forward(images, sample=False)

                loss = F.nll_loss(output, targets, reduction='sum')
                acc = (output.max(1)[1] == targets).sum().item()

                total_loss += loss.item()
                total_acc += acc
                total_num += len(targets)

        return total_loss / total_num, total_acc / total_num



    def custom_regularization(self, 
                              model_old: nn.Module, 
                              model: nn.Module, 
                              minibatch_size: int, 
                              loss: torch.Tensor = None
                              ) -> torch.Tensor:
        """
        Compute the regularization term

        Parameters
        ----------
        model_old: nn.Module
            The old model

        model: nn.Module
            The current model

        minibatch_size: int
            The size of the minibatch

        loss: torch.Tensor
            The loss

        Returns
        -------
        loss: torch.Tensor
            The loss with regularization
        """
        sigma_weight_reg_sum = 0
        mu_weight_reg_sum = 0
        mu_bias_reg_sum = 0

        # Regularization coefficient
        alpha = 0  # First task then no regularization
        if self.saved:  # If the model is saved
            alpha = 1

        # Loop over the layers
        for (_, save_layer), (_, train_layer) in zip(model_old.named_children(), model.named_children()):
            if not isinstance(train_layer, BayesianLinear) and not isinstance(train_layer, BayesianConv2d):
                continue

            # Calculate mu regularization
            train_weight_mu = train_layer.weight_mu  
            train_bias = train_layer.bias
            save_weight_mu = save_layer.weight_mu  
            save_bias = save_layer.bias

            fan_in, fan_out = _calculate_fan_in_and_fan_out(train_weight_mu)

            train_weight_sigma = torch.log1p(torch.exp(train_layer.weight_rho))  # sigma(l)(t)
            save_weight_sigma = torch.log1p(torch.exp(save_layer.weight_rho))  # sigma(l)(t-1)

            if isinstance(train_layer, BayesianLinear):
                std_init = math.sqrt((2 / fan_in) * self.args['ratio'])
            elif isinstance(train_layer, BayesianConv2d):
                std_init = math.sqrt((2 / fan_out) * self.args['ratio'])

            save_weight_strength = std_init / save_weight_sigma  # 1 / sigma(l)(t-1)

            # Reshape the strength
            if len(save_weight_mu.shape) == 4:  # Convolutional layer
                out_features, in_features, _, _ = save_weight_mu.shape
                L2_strength = save_weight_strength.expand(out_features, in_features, 1, 1)
            else:  # Linear layer
                out_features, in_features = save_weight_mu.shape
                L2_strength = save_weight_strength.expand(out_features,in_features)
            
            bias_strength = torch.squeeze(save_weight_strength)
        
            # Calculate the L2-regularization of term(a)
            mu_weight_reg = (L2_strength * (train_weight_mu - save_weight_mu)).norm(2) ** 2
            mu_bias_reg = (bias_strength * (train_bias - save_bias)).norm(2) ** 2

            # Sum the regularization term(a)
            mu_weight_reg_sum = mu_weight_reg_sum + mu_weight_reg
            mu_bias_reg_sum = mu_bias_reg_sum + mu_bias_reg
            
            # Calculate the regularization of term(b)
            weight_sigma = (train_weight_sigma ** 2 / save_weight_sigma ** 2)  # sigma(l)(t)^2 / sigma(l)(t-1)^2
            sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()

            
        # elbo loss
        loss = loss / minibatch_size
        # L2 loss
        loss = loss + alpha * (mu_weight_reg_sum + mu_bias_reg_sum) / (2 * minibatch_size)
        # sigma regularization
        loss = loss + alpha * sigma_weight_reg_sum / (2 * minibatch_size)
            
        return loss