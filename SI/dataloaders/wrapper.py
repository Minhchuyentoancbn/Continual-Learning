import os
import torch
from typing import List


class CacheClassLabel(torch.utils.data.Dataset):
    """A dataset wrapper that has a quick access to all labels of data."""""

    def __init__(self, dataset: torch.utils.data.Dataset, name=None):
        """
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
        """

        super(CacheClassLabel, self).__init__()
        self.dataset = dataset
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        label_cache_filename = os.path.join(dataset.root, name + '_' + str(len(dataset)) + '.pth')

        if os.path.exists(label_cache_filename):
            self.labels = torch.load(label_cache_filename)
        else:
            for i, data in enumerate(dataset):
                self.labels[i] = data[1]
            torch.save(self.labels, label_cache_filename)
        
        self.number_classes = len(torch.unique(self.labels))


    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target


    def __len__(self):
        return len(self.dataset)
    

class AppendName(torch.utils.data.Dataset):
    """A dataset wrapper that also return the name of dataset/task --> (img, target, name)"""

    def __init__(self, dataset: torch.utils.data.Dataset, name: str, first_class_ind=0):
        """
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
        name : str
        first_class_ind : int (default: 0)
        """

        super(AppendName, self).__init__()
        self.dataset = dataset
        self.name = name
        self.first_class_ind = first_class_ind  # for remapping the class labels


    def __getitem__(self, index):
        img, target = self.dataset[index]
        target = target + self.first_class_ind
        return img, target, self.name


    def __len__(self):
        return len(self.dataset)
    


class Subclass(torch.utils.data.Dataset):
    """
    A dataset wrapper that return the task name and remove the offset of labels (Let the labels start from 0)
    """

    def __init__(self, dataset: CacheClassLabel, class_list: List[int], remap=True):
        """
        Parameters
        ----------
        dataset : CacheClassLabel
        class_list : List[int]
        remap : bool (default: True)
        """

        super(Subclass,self).__init__()
        assert isinstance(dataset, CacheClassLabel), 'dataset must be wrapped by CacheClassLabel'  # Check parent class

        self.dataset = dataset
        self.class_list = class_list
        self.remap = remap
        self.indices = []

        for c in class_list:
            self.indices.extend((dataset.labels == c).nonzero().flatten().tolist())

        if remap:
            self.class_mapping = {c: i for i, c in enumerate(class_list)}


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, index):
        img, target = self.dataset[self.indices[index]]
        if self.remap:
            raw_target = target.item() if isinstance(target,torch.Tensor) else target
            target = self.class_mapping[raw_target]

        return img, target
    

class Permutation(torch.utils.data.Dataset):
    """
    A dataset wrapper that permute the position of features
    """

    def __init__(self, dataset: torch.utils.data.Dataset, permutation: List[int]):
        """
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
        permutation : List[int]
        """

        super(Permutation, self).__init__()
        self.dataset = dataset
        self.permutation = permutation


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        img, target = self.dataset[index]
        shape = img.size()
        img = img.view(-1)[self.permutation].view(shape)
        return img, target



class Storage(torch.utils.data.Subset):
    def reduce(self, m):
        self.indices = self.indices[:m]