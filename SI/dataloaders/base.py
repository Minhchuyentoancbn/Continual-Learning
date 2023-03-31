import torchvision
from torchvision import datasets, transforms
from .wrapper import CacheClassLabel


def MNIST(dataroot):
    """Load MNIST dataset."""

    val_transform = transforms.ToTensor()
    train_transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        dataroot, 
        train=True, 
        download=True,
        transform=train_transform
    )
    train_set = CacheClassLabel(train_set, 'mnist')

    val_set = datasets.MNIST(
        dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_set = CacheClassLabel(val_set, 'mnist')

    return train_set, val_set


