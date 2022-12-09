#!/usr/bin/python3

# pylint: disable=no-member
import torch

from models import Encoder, Decoder
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, ToPILImage
from argparse import ArgumentParser
from code import interact
from PIL import Image


lr = 0.001
alpha = 0.1
beta = 1

dataset = CIFAR10("data", download=True, transform=ToTensor())

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, prefetch_factor=16)

encoder = Encoder()
decoder = Decoder()

if torch.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()

for epoch in range(50):
    if epoch % 10 == 0:
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr) 
        lr *= 0.25

    for batch, label in dataloader:
        if torch.cuda.is_available():
            batch = batch.cuda()

        optimizer.zero_grad()

        mu, sigma = torch.split(encoder.forward(batch), 256, dim=1)
        mu = torch.reshape(mu, (mu.shape[0], mu.shape[1], 1, 1))
        sigma = torch.reshape(sigma, (sigma.shape[0], sigma.shape[1], 1, 1))

        norm_sample = torch.normal(torch.zeros_like(sigma), 0.1 * torch.ones_like(sigma))
        #norm_sample = torch.ones_like(sigma)
        #batch_prime = decoder.forward((sigma * norm_sample) + mu)
        batch_prime = decoder.forward((sigma + norm_sample) + mu)
        
        loss_kl = (sigma - sigma.exp() - mu.square() + 1).mean().mul(-0.5)
        loss_recon = (batch_prime - batch).square().mean()

        loss = alpha * loss_kl + \
               beta * loss_recon

        loss.backward()
        optimizer.step()

        print(f"{epoch} {loss.item():.5e} {loss_kl.item():.5e} {loss_recon.item():.5e}")

decoder.eval()
encoder.eval()

mu, sigma = torch.split(encoder.forward(batch), 256, dim=1)
mu = torch.reshape(mu, (mu.shape[0], mu.shape[1], 1, 1))
sigma = torch.reshape(sigma, (sigma.shape[0], sigma.shape[1], 1, 1))

#norm_sample = torch.normal(torch.zeros_like(sigma), 0.01 * torch.ones_like(sigma))
norm_sample = torch.ones_like(sigma)
imgs = decoder.forward((sigma * norm_sample) + mu)

for idx, img in enumerate(imgs):
    ToPILImage()(img).save(f"/tmp/imgs/batch_{idx}.png")


norm_sample = torch.normal(torch.zeros(64, 256, 1, 1), torch.ones(64, 256, 1, 1))
if torch.cuda.is_available():
    norm_sample = norm_sample.cuda()

imgs = decoder.forward(norm_sample)

for idx, img in enumerate(imgs):
    ToPILImage()(img).save(f"/tmp/imgs/img_{idx}.png")

interact(local=locals())
