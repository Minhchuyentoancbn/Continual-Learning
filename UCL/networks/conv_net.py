import torch
import torch.nn as nn
from utils import *
from typing import Union, List

class Net(nn.Module):
    """Convolutional neural network for CIFAR 10/100 dataset."""

    def __init__(self, inputsize: Union[int, int, int], taskcla: List[Union[int, int]]):
        """
        Parameters
        ----------
        inputsize : Union[int, int, int]
            Size of the input image.

        taskcla : List[Union[int, int]]
            List of the number of classes for each task.
        """
        super().__init__()
        
        ncha, size, _ = inputsize
        self.taskcla = taskcla
        
        self.conv1 = nn.Conv2d(ncha, 32, kernel_size=3, padding=1)
        s = compute_conv_output_size(size, 3, padding=1) # 32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s // 2 # 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
#         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
#         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = nn.Linear(s*s*128,256) # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256, n))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        h=self.relu(self.conv1(x))
        h=self.relu(self.conv2(h))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv3(h))
        h=self.relu(self.conv4(h))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv5(h))
        h=self.relu(self.conv6(h))
#         h=self.relu(self.conv7(h))
        h=self.drop1(self.MaxPool(h))
        h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h)))
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        return y