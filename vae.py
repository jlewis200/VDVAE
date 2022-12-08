#!/usr/bin/python3

# pylint: disable=no-member
import torch

from models import Encoder, Decoder
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from argparse import ArgumentParser
from code import interact
from PIL import Image


lr = 0.01
alpha = 1
beta = 1

dataset = CIFAR10("data", download=True, transform=ToTensor())

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, prefetch_factor=16)

encoder = Encoder()
decoder = Decoder()
#optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr) 
optimizer = torch.optim.Adam(encoder.parameters(), lr=lr) 

if torch.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()

for epoch in range(10):
    lr *= 0.3

    for batch, _ in dataloader:
        if torch.cuda.is_available():
            batch = batch.cuda()

        optimizer.zero_grad()

        mu, sigma = torch.split(encoder.forward(batch), 256, dim=1)
        mu = torch.reshape(mu, (mu.shape[0], mu.shape[1], 1, 1))
        sigma = torch.reshape(sigma, (sigma.shape[0], sigma.shape[1], 1, 1))

        norm_sample = torch.normal(torch.zeros_like(sigma), torch.ones_like(sigma))
        batch_prime = decoder.forward((sigma * norm_sample) + mu)

        loss_kl = (sigma.square().log() - sigma.square() - mu.square() + 1).sum().mul(0.5).div(sigma.numel())
        loss_recon = (batch_prime - batch).square().sum().div(batch.numel())

        loss = -alpha * loss_kl + \
               beta * loss_recon

        loss.backward()
        optimizer.step()

        print(f"{epoch} {loss.item():.5e} {loss_kl.item():.5e} {loss_recon.item():.5e}")

interact(local=locals())
