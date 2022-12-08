#!/usr/bin/python3

"""
"""

import torch

from torch import nn
from code import interact


class Decoder(nn.Sequential):
    def __init__(self):
        super(Decoder, self).__init__()

        kernel_size = (3, 3)
        padding = 0
        output_padding = 0
        padding_mode = "reflect"
        leaky_relu_lower = 0.0
        flat_size = 256
        
        #shape:  N x 256 x 1 x 1
        self.append(nn.ConvTranspose2d(256, 128, kernel_size=2, padding=padding, output_padding=output_padding))
        #shape:  N x 128 x 4 x 4
        self.append(nn.ConvTranspose2d(128, 64, kernel_size=3, padding=padding, output_padding=output_padding))
        #shape:  N x 256 x 4 x 4
        self.append(nn.ConvTranspose2d(64, 32, kernel_size=5, padding=padding, output_padding=output_padding))
        #shape:  N x 256 x 8 x 8
        self.append(nn.ConvTranspose2d(32, 16, kernel_size=9, padding=padding, output_padding=output_padding))
        #shape:  N x 256 x 16 x 16
        self.append(nn.ConvTranspose2d(16, 3, kernel_size=17, padding=padding, output_padding=output_padding))
        #shape:  N x 256 x 32 x 32


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        kernel_size = (3, 3)
        padding = 1
        padding_mode = "reflect"
        leaky_relu_lower = 0.0
        flat_size = 256
        
        self.shared = nn.Sequential()
        self.mu = nn.Sequential()
        self.sigma = nn.Sequential()
        #shapes shown as N x C x H x W
        #N:  number of samples in this batch (number of "flows")
        #C:  number of channels
        #H:  height of "flow" matrix
        #W:  width of "flow" matrix

        #in:  N x 3 x 32 x 32
        self.shared.append(nn.Conv2d(3, 16, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.Conv2d(16, 16, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.MaxPool2d((2, 2)))
        #out:  N x 32 x 16 x 16

        #in:  N x 32 x 16 x 16
        self.shared.append(nn.Conv2d(16, 32, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.Conv2d(32, 32, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.MaxPool2d((2, 2)))
        #out:  N x 64 x 8 x 8

        #in:  N x 64 x 8 x 8
        self.shared.append(nn.Conv2d(32, 64, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.Conv2d(64, 64, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.MaxPool2d((2, 2)))
        #out:  N x 128 x 4 x 4

        #in:  N x 128 x 4 x 4
        self.shared.append(nn.Conv2d(64, 128, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.Conv2d(128, 128, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.MaxPool2d((2, 2)))
        #out:  N x 256 x 2 x 2

        #in:  N x 128 x 2 x 2
        self.shared.append(nn.Conv2d(128, 256, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.Conv2d(256, 256, kernel_size, padding=padding, padding_mode=padding_mode))
        self.shared.append(nn.LeakyReLU(leaky_relu_lower))
        self.shared.append(nn.MaxPool2d((2, 2)))
        #out:  N x 256 x 1 x 1

        #in:  N x 256 x 1 x 1
        self.shared.append(nn.Flatten())
        self.shared.append(nn.BatchNorm1d(flat_size))
        #out: N x 256

        #in: N x 256
        self.mu.append(nn.Linear(flat_size, flat_size))
        self.mu.append(nn.Linear(flat_size, flat_size))
        #out: N x 256

        #in: N x 256
        self.sigma.append(nn.Linear(flat_size, flat_size))
        self.sigma.append(nn.Linear(flat_size, flat_size))
        #out: N x 256

    def forward(self, x):
        intermediate = self.shared.forward(x)
        mu = self.mu.forward(intermediate)
        sigma = self.sigma.forward(intermediate)

        #out N x 512
        return torch.cat((mu, sigma), dim=1)
