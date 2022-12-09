#!/usr/bin/python3

"""
"""

import torch

from torch import nn
from code import interact


class Decoder(nn.Sequential):
    def __init__(self):
        super(Decoder, self).__init__()

        kernel_size = 3
        stride = 2
        padding = 0
        output_padding = 0
        padding_mode = "reflect"
        negative_slope = 0.0
        flat_size = 1024 
        
        #shape:  N x 256 x 1 x 1
        self.append(nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))
        self.append(nn.BatchNorm2d(128))
        self.append(nn.LeakyReLU(negative_slope))

        #shape:  N x 128 x 3 x 3
        self.append(nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))
        self.append(nn.BatchNorm2d(64))
        self.append(nn.LeakyReLU(negative_slope))
        
        #shape:  N x 64 x 7 x 7
        self.append(nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))
        self.append(nn.BatchNorm2d(32))
        self.append(nn.LeakyReLU(negative_slope))
        
        #shape:  N x 32 x 15  x 15
        self.append(nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))
        self.append(nn.BatchNorm2d(16))
        self.append(nn.LeakyReLU(negative_slope))
        
        #shape:  N x 16 x 31 x 31
#        self.append(nn.ConvTranspose2d(16, 3, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))
#        self.append(nn.BatchNorm2d(3))
#        self.append(nn.LeakyReLU(negative_slope))
        
        #shape:  N x 256 x 32 x 32

        self.append(nn.Conv2d(16, 3, kernel_size=4, stride=1, padding=2))
        self.append(nn.Sigmoid())


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        kernel_size = 3
        stride = 2
        padding = 1
        negative_slope = 0.01
        momentum = 0.1
        affine = True
        bias = True
        flat_size = 1024 
        
        self.shared = nn.Sequential()
        self.mu = nn.Sequential()
        self.sigma = nn.Sequential()
        #shapes shown as N x C x H x W
        #N:  number of samples in this batch (number of "flows")
        #C:  number of channels
        #H:  height of "flow" matrix
        #W:  width of "flow" matrix

        #in:  N x 3 x 32 x 32
        self.shared.append(nn.Conv2d(3, 32, kernel_size, padding=padding, stride=stride))
        self.shared.append(nn.BatchNorm2d(32, momentum=momentum, affine=affine))
        self.shared.append(nn.LeakyReLU(negative_slope))
        #out:  N x 32 x 16 x 16

        #in:  N x 64 x 8 x 8
        self.shared.append(nn.Conv2d(32, 64, kernel_size, padding=padding, stride=stride))
        self.shared.append(nn.BatchNorm2d(64, momentum=momentum, affine=affine))
        self.shared.append(nn.LeakyReLU(negative_slope))
        #out:  N x 128 x 4 x 4

        #in:  N x 128 x 4 x 4
        self.shared.append(nn.Conv2d(64, 128, kernel_size, padding=padding, stride=stride))
        self.shared.append(nn.BatchNorm2d(128, momentum=momentum, affine=affine))
        self.shared.append(nn.LeakyReLU(negative_slope))
        #out:  N x 256 x 2 x 2

        #in:  N x 128 x 2 x 2
        self.shared.append(nn.Conv2d(128, 256, kernel_size, padding=padding, stride=stride))
        self.shared.append(nn.BatchNorm2d(256, momentum=momentum, affine=affine))
        self.shared.append(nn.LeakyReLU(negative_slope))
        #out:  N x 256 x 1 x 1

        #in:  N x 256 x 1 x 1
        self.shared.append(nn.Flatten())
        #out: N x 256

        #in: N x 256
        self.mu.append(nn.Linear(flat_size, 256, bias=bias))
        #out: N x 256

        #in: N x 256
        self.sigma.append(nn.Linear(flat_size, 256, bias=bias))
        #out: N x 256

    def forward(self, x):
        intermediate = self.shared.forward(x)
        mu = self.mu.forward(intermediate)
        sigma = self.sigma.forward(intermediate)

        #out N x 512
        return torch.cat((mu, sigma), dim=1)
