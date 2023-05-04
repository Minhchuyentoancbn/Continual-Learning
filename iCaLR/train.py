import time
import copy
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
nb_protos  = 20        # Number of prototypes per class at the end: total protoset memory/ total number of classes
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
order = np.arange(100)
np.random.shuffle(order)
np.save('order', order)

# Initialization
dictionary_size     = 500 - nb_val
top1_acc_list_cumul = np.zeros((int(100 / nb_cl), ))

print()
print('Starting Task Incremental Learning...')

# Build the neural network
device    = torch.device("cuda:0" if torch.cuda.is_available() and device=='gpu' else "cpu")
network   = resnet32()
optimizer = optim.SGD(network.parameters(), lr=lr_old, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_strat, gamma=1.0 / lr_factor)
ce_loss   = nn.CrossEntropyLoss()
bce_loss  = nn.BCELoss()

# Initialization of the variables
X_valid_cumuls    = []
X_protoset_cumuls = []
X_train_cumuls    = []
y_valid_cumuls    = []
y_protoset_cumuls = []
y_train_cumuls    = []
alpha_dr_herding  = np.zeros((100 // nb_cl, dictionary_size, nb_cl), np.float32)


# The following contains all the training samples of the different classes 
# because we want to compare our method with the theoretical case where all the training samples are stored
prototypes = torch.zeros(100, dictionary_size, X_train_total.shape[1], X_train_total.shape[2], X_train_total.shape[3])
for orde in range(100):
    prototypes[orde, :, :, :, :] = X_train_total[y_train_total == orde]
# prototypes = prototypes.to(device)

# Iterate through each group
for iteration in range(int(100 / nb_cl)):
    network = network.to(device)
    # Save the results at each increment
    np.save('top1_acc_list_cumul_icarl_cl' + str(nb_cl), top1_acc_list_cumul)

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

    # Add the stored exemplars to the training data
    if iteration > 0:
        X_protoset = torch.cat(X_protoset_cumuls)
        y_protoset = torch.cat(y_protoset_cumuls)
        X_train    = torch.cat((X_train, X_protoset))
        y_train    = torch.cat((y_train, y_protoset))

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
            loss = bce_loss(outputs, targets)
            # Compute L2-regularization loss
            l2_reg = 0
            for param in network.parameters():
                l2_reg += torch.norm(param) ** 2
            loss += l2_reg * wght_decay

            train_err += loss.item()

            # Distillation loss (for previous classes)
            if iteration > 0:
                with torch.no_grad():
                    prediction_old = network_old(inputs)[0]
                targets[:, old_cl] = prediction_old[:, old_cl]
                distillation_loss  = bce_loss(outputs, targets)

                # L2-regularization loss
                # NOTE: I don't know whether it is necessary to add the L2-regularization loss to the distillation loss
                l2_reg_distill = 0
                for param in network.parameters():
                    l2_reg_distill += torch.norm(param) ** 2
                distillation_loss += l2_reg_distill * wght_decay
                loss += distillation_loss

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
    torch.save(network.state_dict(), 'network' + str(iteration + 1) + '_of_' + str(int(100 / nb_cl)) + '.pt')

    # Examplars selection
    nb_protos_cl = int(np.ceil(nb_protos * 100.0 / nb_cl / (iteration + 1)))  # Number of exemplars per class

    # Herding
    print('Updating exemplar set...')
    network.eval()
    
    # Compute rank of the potential exemplars for each class of the current class batch
    for iter_dico in range(nb_cl):
        # Possible exemplars in the feature space and projected on the L2 sphere
        with torch.no_grad():
            mapped_prototypes = network(prototypes[iteration * nb_cl + iter_dico].float().to(device))[1].cpu().numpy()  # Get the feature map of the prototypes of each class
        D = mapped_prototypes.T  # (K x N)
        D = D / np.linalg.norm(D, axis=0)  # L2 normalization

        # Herding procedure : ranking of the potential exemplars
        mu = np.mean(D, axis=1)  # (K, )
        alpha_dr_herding[iteration, :, iter_dico] = 0  # Current class
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not(np.sum(alpha_dr_herding[iteration, :, iter_dico] != 0) == min(nb_protos_cl, 500)) and iter_herding_eff < 1000:  # Check that we have the right number of exemplars
            tmp_t = np.dot(w_t, D)  # (N, ), dot product with mean of the class
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if alpha_dr_herding[iteration, ind_max, iter_dico] == 0:
                alpha_dr_herding[iteration, ind_max, iter_dico] = 1 + iter_herding  # Rank of the selected exemplars
                iter_herding += 1
            w_t = w_t + mu - D[:,ind_max]

    # Prepare the protoset
    X_protoset_cumuls = []
    y_protoset_cumuls = []

    # Class means for iCaRL + Storing the selected exemplars in the protoset for all seen classes
    print('Computing mean-of_exemplars and theoretical mean...')
    class_means = np.zeros((64, 100))
    for iteration2 in range(iteration + 1):
        for iter_dico in range(nb_cl):
            current_cl = torch.from_numpy(order[range(iteration2 * nb_cl, (iteration2 + 1) * nb_cl)])

            # Collect data in the feature space for each class
            with torch.no_grad():
                mapped_prototypes  = network(prototypes[iteration2 * nb_cl + iter_dico].float().to(device))[1].cpu().numpy()
                # mapped_prototypes2 = network(prototypes[iteration2 * nb_cl + iter_dico].flip(-1).float())[1].cpu().numpy()
            D = mapped_prototypes.T  # (K x N)
            D = D / np.linalg.norm(D, axis=0)  # L2 normalization

            # # Flipped version also  
            # # NOTE: I don't know if this is necessary
            # D2 = mapped_prototypes2.T  # (K x N)
            # D2 = D2 / np.linalg.norm(D2, axis=0)  # L2 normalization

            # iCaRL
            alph = alpha_dr_herding[iteration2, :, iter_dico]
            alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.  # Select the best protoset
            X_protoset_cumuls.append(prototypes[iteration2 * nb_cl + iter_dico, np.where(alph == 1)[0]])
            y_protoset_cumuls.append(current_cl[iter_dico] * torch.ones(len(np.where(alph == 1)[0])))
            alph = alph / np.sum(alph)  # Normalize the ranks
            class_means[:, current_cl[iter_dico]] = np.dot(D, alph)  # (np.dot(D, alph) + np.dot(D2, alph)) / 2
            class_means[:, current_cl[iter_dico]] /= np.linalg.norm(class_means[:, current_cl[iter_dico]])

    np.save('cl_means', class_means)
    # Calculate validation error of model on the cumul of classes:
    print('Computing cumulative accuracy...')
    
    stat_icarl = []

    for batch in iterate_minibatches(X_valid_cumul, y_valid_cumul, min(500, len(X_valid)), shuffle=False):
        inputs, targets_prep = batch
        inputs  = inputs.to(device)
        targets = torch.zeros(inputs.shape[0], 100, dtype=torch.float32)
        targets[range(len(targets_prep)), targets_prep.long()] = 1

        network.eval()
        with torch.no_grad():
            outputs, features = network(inputs)
            outputs = outputs.cpu()
            features = features.cpu()
            # Normalize
            features = features / features.norm(dim=1)[:, None]
            features = features.numpy()

            # Compute score for iCaRL
            sqd = cdist(class_means.T, features, 'sqeuclidean')
            score_icarl = (-sqd).T

            stat_icarl.append(np.mean(np.argmax(score_icarl, axis=1) == targets_prep.numpy()))
        
    print('  cumulative accuracy: \t\t\t{:.2f} %'.format(np.mean(stat_icarl) * 100))
    top1_acc_list_cumul[iteration] = np.mean(stat_icarl) * 100
    
    network = network.cpu()

    # Empty the cache
    torch.cuda.empty_cache()