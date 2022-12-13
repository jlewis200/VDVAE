#!/usr/bin/python3

"""
"""

import torch

from torch import nn
from code import interact


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view((-1, *self.shape))


class Decoder(nn.Sequential):
    def __init__(self):
        super(Decoder, self).__init__()

        kernel_size = 3
        stride = 2
        padding = 1
        output_padding = 1
        padding_mode = "reflect"
        negative_slope = 0.0
        bias = True
        flat_size = 4096 
        
        #shape:  N x 1024 

        self.append(nn.Linear(1024, flat_size, bias=bias))
        self.append(nn.Dropout(p=0.2))
        self.append(nn.Linear(flat_size, flat_size, bias=bias))
        
        #shape:  N x 4096

        self.append(View((1024, 2, 2)))
        
        #shape:  N x 1024 x 2 x 2

        in_size = 1024
        out_size = 512

        for _ in range(6):
            self.append(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            self.append(nn.BatchNorm2d(out_size))
            self.append(nn.LeakyReLU(negative_slope))
            
            in_size = out_size
            out_size //= 2
            out_size = max(32, out_size)

        #shape:  N x 32 x 128 x 128

        self.append(nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1))
        
        #shape:  N x 3 x 128 x 128
        
        self.append(nn.Sigmoid())


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        eps=1e-5
        kernel_size = 3
        stride = 2
        padding = 1
        negative_slope = 0.01
        momentum = 0.1
        affine = True
        bias = True
       
        out_size = 1024
        flat_size = 4096

        
        self.shared = nn.Sequential()
        self.mu = nn.Sequential()
        self.sigma = nn.Sequential()

        in_channels = 3
        out_channels = 32 

        #shape:  N x 3 x 128 x 128

        for _ in range(6):
            self.shared.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
            self.shared.append(nn.BatchNorm2d(out_channels, momentum=momentum, affine=affine))
            self.shared.append(nn.LeakyReLU(negative_slope))

            in_channels = out_channels
            out_channels *= 2

        #shape:  N x 1024 x 2 x 2
        
        self.shared.append(nn.Flatten())
        self.shared.append(nn.Linear(flat_size, flat_size, bias=bias))
        self.shared.append(nn.Dropout(p=0.2))
        
        #shape: N x 4096

        #in: N x  2048
        self.mu.append(nn.Linear(flat_size, out_size, bias=bias))
        #out: N x 256

        #in: N x  2048
        self.sigma.append(nn.Linear(flat_size, out_size, bias=bias))
        #out: N x 256

    def forward(self, x):
        intermediate = self.shared.forward(x)
        mu = self.mu.forward(intermediate)
        sigma = self.sigma.forward(intermediate)

        #out (N x 256, N x 256)
        return mu, sigma
