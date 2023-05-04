import time
import copy
import os
import numpy as np

from scipy.spatial.distance import cdist
from utils_cifar100 import iterate_minibatches, load_data
from resnet import resnet32

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters Setting
batch_size = 128
nb_val     = 0         # Number of validation samples per class
nb_cl      = 10        # Number of classes per group
epochs     = 70
lr_old     = 2.0       # Initial learning rate
lr_strat   = [49, 63]  # Epochs where learning rate gets decreased
lr_factor  = 5.0       # Learning rate decrease factor
wght_decay = 1e-5      # Weight Decay
device     = 'gpu'


# Seed everything
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Load data
print('Loading data...')
data = load_data(nb_val)
X_train_total = data['X_train']
y_train_total = data['y_train']

if nb_val != 0:
    X_valid_total = data['X_valid']
    y_valid_total = data['y_valid']
else:
    X_valid_total = data['X_test']
    y_valid_total = data['y_test']


 # Select the order for the class learning
if not os.path.exists('order.npy'):
    order = np.arange(100)
    np.random.shuffle(order)
    np.save('order', order)
else:
    order = np.load('order.npy')

# Initialization
top1_acc_list_cumul = np.zeros((int(100 / nb_cl), ))

print()
print('Starting Task Incremental Learning...')

# Build the neural network
device    = torch.device("cuda:0" if torch.cuda.is_available() and device=='gpu' else "cpu")
network   = resnet32().to(device)
bce_loss  = nn.BCELoss()

# Initialization of the variables
X_valid_cumuls    = []
X_train_cumuls    = []
y_valid_cumuls    = []
y_train_cumuls    = []


# Iterate through each group
for iteration in range(int(100 / nb_cl)):
    # Reset the optimizer at the beginning of each loop
    optimizer = optim.SGD(network.parameters(), lr=lr_old, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_strat, gamma=1.0 / lr_factor)


    # Prepare the training data for the current batch of classes
    actual_cl        = order[range(iteration * nb_cl, (iteration + 1) * nb_cl)]
    old_cl           = order[range(0, iteration * nb_cl)]
    indices_train_10 = np.isin(y_train_total, actual_cl)
    indices_test_10  = np.isin(y_valid_total, actual_cl)
    X_train          = X_train_total[indices_train_10]
    X_valid          = X_valid_total[indices_test_10]
    X_train_cumuls.append(X_train)
    X_valid_cumuls.append(X_valid)
    X_train_cumul    = torch.cat(X_train_cumuls)
    X_valid_cumul    = torch.cat(X_valid_cumuls)
    y_train          = y_train_total[indices_train_10]
    y_valid          = y_valid_total[indices_test_10]
    y_train_cumuls.append(y_train)
    y_valid_cumuls.append(y_valid)
    y_train_cumul    = torch.cat(y_train_cumuls)
    y_valid_cumul    = torch.cat(y_valid_cumuls)

    # The training loop
    print()
    print(f'Batch of classes number {iteration + 1} arrives ...')

    for epoch in range(epochs):
        # Shuffle the training data
        train_indices = np.arange(len(X_train))
        np.random.shuffle(train_indices)
        X_train       = X_train[train_indices]
        y_train       = y_train[train_indices]

        # In each epoch, we do a full pass over the training data:
        train_batches = 0
        train_err     = 0
        start_time    = time.time()
        
        network.train()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True, augment=True):
            inputs, targets_prep = batch
            inputs = inputs.to(device)
            targets = torch.zeros(inputs.shape[0], 100, dtype=torch.float32)
            targets[range(len(targets_prep)), targets_prep.long()] = 1
            targets = targets.to(device)

            outputs, intermeds = network(inputs)

            # Classification loss (only for the new classes)
            if iteration > 0:  # Distillation
                with torch.no_grad():
                    prediction_old = network_old(inputs)[0]
                targets[:, old_cl] = prediction_old[:, old_cl]

            loss = bce_loss(outputs, targets)

            # Compute L2-regularization loss
            l2_reg = 0
            for param in network.parameters():
                l2_reg += torch.norm(param) ** 2
            loss += l2_reg * wght_decay

            train_err += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_batches % 100 == 0:
                print(f'Loss: {loss.item():.3f}')

            train_batches += 1

        scheduler.step()
        
        # A full pass over the validation data
        val_err     = 0
        top1_acc    = 0
        val_batches = 0
        network.eval()

        with torch.no_grad():
            for batch in iterate_minibatches(X_valid, y_valid, min(500, len(X_valid)), shuffle=False):
                inputs, targets_prep = batch
                inputs  = inputs.to(device)
                targets = torch.zeros(inputs.shape[0], 100, dtype=torch.float32)
                targets[range(len(targets_prep)), targets_prep.long()] = 1

                outputs, intermeds = network(inputs)
                outputs = outputs.cpu()
                intermeds = intermeds.cpu()
                val_err += bce_loss(outputs, targets).item()  # Classification loss (only for the new classes)
                
                # Top-1 accuracy
                _, predicted = torch.max(outputs, 1)
                top1_acc += (predicted == targets_prep).sum().item() / inputs.shape[0]

                val_batches += 1

        top1_acc  /= val_batches
        val_err   /= val_batches
        train_err /= train_batches
        print(f'Batch of classes {iteration + 1} out of {int(100 / nb_cl)} batches')
        print('Epoch {} took {:.2f}s'.format(epoch + 1, time.time() - start_time), end='')
        print('||  training loss (in-iteration): \t{:.6f}'.format(train_err), end='')
        print('||  validation loss (in-iteration): \t{:.6f}'.format(val_err), end='')
        print('||  validation accuracy: \t\t\t{:.2f} %'.format(top1_acc * 100))

    # Duplicate current network to distill later
    if iteration == 0:
        network_old = resnet32().to(device)
    
    network_old.load_state_dict(copy.deepcopy(network.state_dict()))
    # Save the network
    torch.save(network.state_dict(), 'models/network' + str(iteration + 1) + '_of_' + str(int(100 / nb_cl)) + '_lwfmc.pt')

    network.eval()
    # Calculate validation error of model on the cumul of classes:
    print('Computing cumulative accuracy...')
    
    stat_lwfmc = []

    for batch in iterate_minibatches(X_valid_cumul, y_valid_cumul, min(500, len(X_valid)), shuffle=False):
        inputs, targets_prep = batch
        inputs  = inputs.to(device)
        targets = torch.zeros(inputs.shape[0], 100, dtype=torch.float32)
        targets[range(len(targets_prep)), targets_prep.long()] = 1

        network.eval()
        with torch.no_grad():
            outputs = network(inputs)[0].cpu()
            _, predicted = torch.max(outputs, 1)
            stat_lwfmc.append((predicted == targets_prep).sum().item() / inputs.shape[0])
        
    print('  cumulative accuracy: \t\t\t{:.2f} %'.format(np.mean(stat_lwfmc) * 100))
    top1_acc_list_cumul[iteration] = np.mean(stat_lwfmc) * 100


    # Save the results at each increment
    np.save('top1_acc_list_cumul_lwfmc_cl' + str(nb_cl), top1_acc_list_cumul)
    
torch.cuda.empty_cache()