import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP."""

    def __init__(self, out_dim=10, in_channel=1, img_size=32, hidden_dim=256):
        """Initialize MLP.

        Parameters
        ----------
        out_dim : int, optional
            output dimension, by default 10

        in_channel : int, optional
            input channel, by default 1

        img_size : int, optional
            image size, by default 32

        hidden_dim : int, optional
            hidden dimension, by default 256

        Returns
        -------
        None
        """
        super().__init__()
        self.in_dim = in_channel * img_size * img_size
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)


    def features(self, x):
        x = self.linear(x.view(-1, self.in_dim))
        return x
    

    def logits(self, x):
        x = self.last(x)
        return x


    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
    

def MLP256_MNIST():
    """Return MLP with 256 hidden units on MNIST."""
    return MLP(hidden_dim=256, img_size=28)