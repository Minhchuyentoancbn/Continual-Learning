import torch
import random
from .default import NormalNN
from typing import Dict


class L2(NormalNN):
    """
    L2 regularization.
    """

    def __init__(self, agent_config: dict):
        """
        Initialize the L2 regularization agent.

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
        super(L2, self).__init__(agent_config)
        self.params = {n : p for n, p in self.model.named_parameters() if p.requires_grad}
        self.regularization_terms = {}  # Regularization terms for each task
        self.task_count = 0  # The number of tasks
        self.online_reg = True  # True: There will be only one importance matrix and previous model parameters
                                # False: Each task has its own importance matrix and model parameters
        
    
    def calculate_importance(self, dataloader: torch.utils.data.DataLoader):
        """
        Calculate the importance of each parameter.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            dataloader

        Returns
        -------
        importance : dict
            importance of each parameter
        """

        # Use an identity importance so it is an L2 regularization.
        importance = {}

        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Idnetity importance
        return importance
    

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
        if self.log:
            print(f'#reg_term: {len(self.regularization_terms)}')

        # 1.Learn the parameters for current task
        super(L2, self).learn_batch(train_loader, val_loader)

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3. Calculate the importance of weights for current task
        importance = self.calculate_importances(train_loader)

       # Save the weight and importance of weights of current task
        self.task_count += 1
        if len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {'importance': importance, 'task_param': task_param}
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {'importance': importance, 'task_param': task_param}


    def criterion(self, inputs: Dict[str, torch.Tensor], targets: torch.Tensor, tasks: torch.Tensor, regularization=True, **kwargs) -> torch.Tensor:
        """
        Calculate the loss.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            The inputs tensor.

        targets : torch.Tensor
            The target tensor.

        tasks : torch.Tensor
            The task tensor.

        regularization : bool
            Whether to use regularization.

        Returns
        -------
        loss : torch.Tensor
            The loss.
        """

        loss = super(L2, self).criterion(inputs, targets, tasks,  **kwargs)

        if regularization and len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0

            # Sum the regularization loss over previous tasks
            for i, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                reg_loss += task_reg_loss
            loss += reg_loss * self.config['reg_coef']

        return loss
    

class EWC(L2):
    """
    Elastic Weight Consolidation.
    """
    
    def __init__(self, agent_config: dict):
        """
        Initialize the EWC agent.

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
        super(EWC, self).__init__(agent_config)
        self.online_reg = False  # True: There will be only one importance matrix and previous model parameters
                                 # False: Each task has its own importance matrix and model parameters
        self.n_fisher_sample = None  # The number of samples for estimating the Fisher information
        self.empFI = False  # True: Use the empirical Fisher information


    def calculate_importance(self, dataloader: torch.utils.data.DataLoader):
        """
        Calculate the importance of each parameter.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            dataloader

        Returns
        -------
        importance : dict
            importance of each parameter
        """

        if self.log:
            print('Computing EWC')

        # Initialize the importance matrix
        if self.online_reg and len(self.regularization_terms)>0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
        
        # Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
        # Otherwise it uses mini-batches for the estimation. This speeds up the process a lot with similar performance.
        if self.n_fisher_sample is not None:
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            if self.log:
                print('Sample', n_sample, 'for estimating the F matrix.')
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, batch_size=1)

        mode = self.training
        self.eval()

        # Accumulate the square of gradients
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()

            preds = self.forward(input)

            # Sample the labels for estimating the gradients
            # For multi-headed model, the batch of data will be from the same task,
            # so we just use task[0] as the task name to fetch corresponding predictions
            # For single-headed model, just use the max of predictions from preds['All']
            task_name = task[0] if self.multihead else 'All'

            # The flag self.valid_out_dim is for handling the case of incremental class learning.
            # if self.valid_out_dim is an integer, it means only the first 'self.valid_out_dim' dimensions are used
            # in calculating the loss.
            pred = preds[task_name] if not isinstance(self.valid_out_dim, int) else preds[task_name][:,:self.valid_out_dim]
            ind = pred.max(1)[1].flatten()  # Choose the one with max

            # - Alternative ind by multinomial sampling. Its performance is similar. -
            # prob = torch.nn.functional.softmax(preds['All'],dim=1)
            # ind = torch.multinomial(prob,1).flatten()

            if self.empFI:  # Use groundtruth label (default is without this)
                ind = target

            loss = self.criterion(preds, ind, task, regularization=False)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += ((self.params[n].grad ** 2) * len(input) / len(dataloader))

        self.train(mode=mode)

        return importance
    

def EWC_online(agent_config):
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
    """
    agent = EWC(agent_config)
    agent.online_reg = True
    return agent


class SI(L2):
    """
    Synaptic Intelligence.
    """
        
    def __init__(self, agent_config: dict):
        """
        Initialize the SI agent.

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
        super(SI, self).__init__(agent_config)
        self.online_reg = True  # True: There will be only one importance matrix and previous model parameters
                                # False: Each task has its own importance matrix and model parameters
        self.damping_factor = 1e-3

        self.w = {}  # Parameters contribution to change in loss
        
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()


    def update_model(self, inputs: torch.Tensor, targets: torch.Tensor, tasks: torch.Tensor):
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
        unreg_gradients = {}

        # 1. Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2. Compute the gradients of the loss w.r.t. the parameters without regularization
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks, regularization=False)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        # 3. Normal update with regularization
        loss = self.criterion(out, targets, tasks, regularization=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Update the parameters contribution to change in loss
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            # In multi-head network, some head could have no grad (lazy) since no loss go through it.
            if n in unreg_gradients.keys():
                self.w[n] -= delta * unreg_gradients[n]

        return loss.detach(), out


    def calculate_importances(self, dataloader):
        """
        Calculate the importance of each parameter.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            dataloader

        Returns
        -------
        importance : dict
            importance of each parameter
        """
        if self.log:
            print('Computing SI')

        assert self.online_reg,'SI needs online_reg=True'

        # Initialize the importance matrix
        if len(self.regularization_terms) > 0:  # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # The case of the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)
            prev_params = self.initial_params

        # Calculate the importance of each parameter
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n] / (delta_theta ** 2 + self.damping_factor)
            self.w[n].zero_()

        return importance


 