'''
This module contains the definition of the class, "ConvNetwork". The class defines a convolutional neural network computation graph using Pytorch. The implementation enables varying the number of layers to be optimized in hyper-parameters tuning.
'''

import torch.nn as nn
import torch

NB_DIGITS = 10 # Number of labels, i.e. 0, 1, 2, ... , 9.
N_PIXELS = 28 # the hand-writing images are wth resolutions at (28 x 28).

class ConvNetwork(nn.Module):
    '''
    This class defines the computation graph.
    '''
    def __init__(self, n_layers=3):
        '''
        Initialize the computation graph.
        '''
        super(ConvNetwork, self).__init__()
        self.size = (N_PIXELS, N_PIXELS)
        self.no_labels = NB_DIGITS

        # 0. Define a function to determine the dimension of the output of CNN layer.
        def nconv(size, kernel_size=5, stride=1):
            return (size-(kernel_size-1)-1) // stride + 1

        # 1. Stack CNN layers "n_layers" times.
        self._h = nconv(self.size[0])
        self._w = nconv(self.size[1])

        modules = []
        modules.append(nn.Conv2d(1, 16, kernel_size=5, stride=1)) # Convolution layer.
        modules.append(nn.BatchNorm2d(16)) # Batch normalization.
        modules.append(nn.ReLU()) # ReLu activation.

        for i in range(n_layers-1):
            self._h = nconv(self._h)
            self._w = nconv(self._w)
            modules.append(nn.Conv2d(16, 16, kernel_size=5, stride=1))
            modules.append(nn.BatchNorm2d(16))
            modules.append(nn.ReLU())
        self.conv = nn.Sequential(*modules) # Stack the list of CNN layers.

        # 2. Define the last dense layer which maps to the distirbutions using a softmax function.
        self.linear = nn.Sequential(
            nn.Linear(self._h*self._w*16, 100),
            nn.Linear(100, self.no_labels),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        X = self.conv(X)
        X = X.view(-1, self._h*self._w*16)
        X = self.linear(X)
        return X



