import torch
from random import shuffle
from .wrapper import AppendName, Subclass, Permutation, CacheClassLabel


def SplitGen(
        train_set: CacheClassLabel, 
        val_set: CacheClassLabel, 
        first_split_size=2, 
        other_split_size=2, 
        rand_split=False, 
        remap_class=False
    ):
    """
    Generate the dataset splits based on the labels.

    Parameters
    ----------
    train_set : CacheClassLabel
        The training dataset.

    val_set : CacheClassLabel
        The validation dataset.

    first_split_size : int, optional
        The size of the first split, by default 2

    other_split_size : int, optional
        The size of the other splits, by default 2

    rand_split : bool, optional
        Whether to randomly split the dataset, by default False

    remap_class : bool, optional
        Whether to remap the classes, by default False

    
    Returns
    -------
    train_loaders: {task_name:loader}
        The training data loaders for each task.

    val_loaders: {task_name:loader}
        The validation data loaders for each task.

    out_dim: {task_name:num_classes}
        The output dimension for each task.
    """
    assert train_set.number_classes == val_set.number_classes,'Train/Val has different number of classes'
    num_classes = train_set.number_classes

    # Calculate the boundary index of classes for splits
    # Ex: [0,2,4,6,8,10] or [0,50,60,70,80,90,100]
    split_boundaries = [0, first_split_size]
    while split_boundaries[-1] < num_classes:
        split_boundaries.append(split_boundaries[-1] + other_split_size)
    print(f"Split boundaries: {split_boundaries}")
    assert split_boundaries[-1] == num_classes,'Invalid split size'


    # Assign classes to each splits
    # Create the dict: {split_name1:[2,6,7], split_name2:[0,3,9], ...}
    if not rand_split:
        class_lists = {str(i): list(range(split_boundaries[i - 1], split_boundaries[i])) for i in range(1, len(split_boundaries))}
    else:
        randseq = torch.randperm(num_classes)
        class_lists = {str(i):randseq[list(range(split_boundaries[i - 1], split_boundaries[i]))].tolist() for i in range(1, len(split_boundaries))}
    print(class_lists)

    # Generate the dicts of splits
    # Ex: {split_name1:dataset_split1, split_name2:dataset_split2, ...}
    train_dataset_splits = {}
    val_dataset_splits = {}
    tasl_ouput_space = {}
    for name, class_list in class_lists.items():
        train_dataset_splits[name] = AppendName(Subclass(train_set, class_list, remap_class), name)
        val_dataset_splits[name] = AppendName(Subclass(val_set, class_list, remap_class), name)
        tasl_ouput_space[name] = len(class_list)

    return train_dataset_splits, val_dataset_splits, tasl_ouput_space


def PermutationGen(
        train_dataset: CacheClassLabel,
        val_dataset: CacheClassLabel,
        num_permutations: int,
        remap_class: bool = False
    ):
    """
    Generate the dataset splits based on the permutations.
    """
    sample, _ = train_dataset[0]
    n = sample.numel()
    train_datasets = {}
    val_datasets = {}
    task_ouput_space = {}

    for i in range(1, num_permutations + 1):
        rand_ind = list(range(n))
        shuffle(rand_ind)
        name = str(i)

        if i == 1:  # First task has no permutation
            train_datasets[name] = AppendName(train_dataset, name)
            val_datasets[name] = AppendName(val_dataset, name)
        else:
            # For incremental tasks, we permute the input
            first_class_ind = (i - 1) * train_dataset.number_classes if remap_class else 0
            train_datasets[name] = AppendName(Permutation(train_dataset, rand_ind), name, first_class_ind=first_class_ind)
            val_datasets[name] = AppendName(Permutation(val_dataset, rand_ind), name, first_class_ind=first_class_ind)
        
        task_ouput_space[name] = train_dataset.number_classes

    return train_datasets, val_datasets, task_ouput_space