#!/usr/bin/python3

"""
Variational Auto Encoder (VAE) model.
"""

import torch
import math
import numpy as np

from torch import nn


class VAE(nn.Module):
    """
    VAE Encoder/Decoder using stride 1 convolutions and avgpool/interpolate for up/down sampling.
    """

    def __init__(self, encoder_layers, decoder_layers):
        super().__init__()

        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers)

    def encode(self, tensor):
        """
        Pass input tensor through the encoder.
        """
    
        return self.encoder.forward(tensor)

    def decode(self, tensor=None):
        """
        Pass latent encoding tensor through the decoder.
        """

        return self.decoder.forward(tensor)

    def sample(self, n_samples, temp=1.0):
        """
        Sample from the model.
        """

        return self.decoder.sample(n_samples, temp)

    def get_loss_kl(self):
        """
        Get the KL divergence loss from the model.
        """

        return self.decoder.get_loss_kl()

class Encoder(nn.Module):
    """
    VAE Encoder class.
    """
    
    def __init__(self, layers):
        super().__init__()

        self.in_conv = nn.Conv2d(3, 512, kernel_size=3, padding=1)

        self.encoder_groups = nn.Sequential()

        for layer in layers:
            self.encoder_groups.append(EncoderGroup(**layer))

    def forward(self, tensor):
        """
        Perform a forward pass through the encoder.
        """

        #input convolution
        tensor = self.in_conv(tensor)

        self.encoder_groups(tensor)

        #collect the activations
        activations = {}

        for encoder_group in self.encoder_groups:
            activations[encoder_group.resolution] = encoder_group.activations

        return activations

class Decoder(nn.Module):
    """
    VAE Decoder class.
    """

    def __init__(self, layers):
        super().__init__()
        
        self.decoder_groups = nn.ModuleList()

        for layer in layers:
            self.decoder_groups.append(DecoderGroup(**layer))

        self.gain = nn.Parameter(torch.ones((1, 512, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, 512, 1, 1)))

        self.out_conv = nn.Conv2d(512, 3, kernel_size=3, padding=1)

    def forward(self, activations):
        """
        Perform a forward pass through the decoder.
        """

        tensor = torch.zeros_like(activations[self.decoder_groups[0].resolution], requires_grad=True)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        for decoder_group in self.decoder_groups:
            tensor = decoder_group(tensor, activations[decoder_group.resolution])

        #tensor = tensor * self.gain + self.bias
        
        return self.out_conv(tensor)

    def sample(self, n_samples, temp=1.0):
        """
        Sample from the model.  Temperature controls the variance of the distributions the latent
        encodings are drawn from.  A higher temperature leads to higher variance and more dynamic
        results.
        """

        shape = (n_samples, self.decoder_groups[0].channels, self.decoder_groups[0].resolution, self.decoder_groups[0].resolution)
        tensor = torch.zeros(shape)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        for decoder_group in self.decoder_groups:
            tensor = decoder_group.sample(tensor, temp=temp)
 
        #tensor = tensor * self.gain + self.bias
        
        return self.out_conv(tensor)       
        
    def get_loss_kl(self):
        """
        Collect the KL divergence loss from all groups.
        """

        losses = []

        #collect the losses
        for decoder_group in self.decoder_groups:
            losses.append(decoder_group.get_loss_kl())

        #sum the tensors (gradients are preserved)
        loss_kl = sum(losses)

        #scale by the original resolution and color depth
        scale_factor = 3 * self.decoder_groups[-1].resolution**2

        loss_kl = loss_kl / scale_factor

        return loss_kl


class EncoderGroup(nn.Module):
    """
    Encoder group encapsulates all encoder blocks at the same tensor H x W.
    """
    
    def __init__(self,
                 channels,
                 resolution,
                 n_blocks,
                 n_downsamples=1):
        super().__init__()

        self.resolution = resolution
        self.n_downsamples = n_downsamples

        #ratio per VDVAE
        mid_channels = channels // 4
        
        self.encoder_blocks = nn.Sequential()

        #use a (3, 3) kernel if the resolution > 1
        mid_kernel = 3 if resolution > 1 else 1
        
        for _ in range(n_blocks):
            self.encoder_blocks.append(Block(channels, mid_channels, channels, mid_kernel=mid_kernel))
        
        #self.encoder_blocks[-1].weight.data *= np.sqrt(1 / n_blocks)
       
    def forward(self, tensor):
        """
        Perform the forward pass through the convolution module.  Store the activations.
        """

        tensor = self.encoder_blocks.forward(tensor)
        
        #store the activations
        self.activations = tensor
        
        #downsample
        for _ in range(self.n_downsamples):
            tensor = nn.functional.avg_pool2d(tensor, 2)
        
        return tensor


class DecoderGroup(nn.Module):
    """
    Decoder group encapsulates all decder blocks at the same tensor H x W.
    """

    def __init__(self,
                 channels,
                 n_blocks,
                 resolution,
                 bias=True,
                 upsample_ratio=2):
        super().__init__()
      
        self.upsample_ratio=upsample_ratio
        self.channels = channels
        self.resolution = resolution
        self.decoder_blocks = nn.ModuleList()

        for _ in range(n_blocks):
            self.decoder_blocks.append(DecoderBlock(channels, resolution))
        
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, channels, resolution, resolution)))

    def forward(self, tensor, activations):
        """
        Perform the forward pass through this group.
        """
        #upsample the output of the previous decoder group
        tensor = nn.functional.interpolate(tensor, scale_factor=self.upsample_ratio)
      
        if self.bias is not None:
            #add the bias term for this decoder group
            tensor = tensor + self.bias
          
        for decoder_block in self.decoder_blocks:
            tensor = decoder_block(tensor, activations)

        return tensor

    def sample(self, tensor, temp=1.0):
        """
        Perform a sampling pass through this group.
        """

        #upsample the output of the previous decoder group
        tensor = nn.functional.interpolate(tensor, scale_factor=self.upsample_ratio)
      
        if self.bias is not None:
            #add the bias term for this decoder group
            tensor = tensor + self.bias
        
        for decoder_block in self.decoder_blocks:
            tensor = decoder_block.sample(tensor, temp)

        return tensor

    def get_loss_kl(self):
        """
        Get the KL divergence loss for each decoder block in this decoder group.
        """

        losses = []

        #gather the KL losses from each decder block
        for decoder_block in self.decoder_blocks:
            losses.append(decoder_block.loss_kl.sum(dim=(1, 2, 3)))

        #sum the tensors (gradients are preserved)
        loss_kl = sum(losses)

        return loss_kl      


class DecoderBlock(nn.Module):
    """
    Decoder block per VDVAE architecture.
    """

    def __init__(self,
                 channels,
                 resolution):
        super().__init__()

        #mid_channels and z_channels ratios from VDVAE
        mid_channels = channels // 4

        #z_channels fixed at 16
        self.z_channels = 16
     
        #use a (3, 3) kernel if the resolution > 1
        mid_kernel = 3 if resolution > 1 else 1

        #block ratios from VDVAE
        self.phi = Block(channels * 2, mid_channels, 2 * self.z_channels, mid_kernel=mid_kernel)
        self.theta = Block(channels, mid_channels, channels + (2 * self.z_channels), mid_kernel=mid_kernel)
        self.z_projection = nn.Conv2d(self.z_channels, channels, kernel_size=1)
        self.res = Block(channels, mid_channels, channels, res=True, mid_kernel=mid_kernel)

    def forward(self, tensor, activations):
        """
        Perform a forward pass through the convolution module.

        Note:  A ValueError during the creation of either normal distribution is a sign the 
        learning rate is too high.  The mins/maxs of p_logvar have been observed to trend toward
        inf/-inf when the LR is too high and training collapses.  The magnitude of the mean/logvar
        tend to increase as the tensor resolution increases.
        """      
       
        #get the mean, log-variance of p, and the residual flow to the main path
        theta = self.theta(tensor)
        p_mean, p_logvar, res = torch.tensor_split(theta, (self.z_channels, 2 * self.z_channels), dim=1)

        #get p = N(p_mean, p_std)
        p_dist = torch.distributions.Normal(p_mean, torch.exp(p_logvar / 2))

        #join the output from theta with the main branch
        tensor = tensor + res

        #get the mean/log-variance of q
        q_mean, q_logvar = self.phi(torch.cat((tensor, activations), dim=1)).chunk(2, dim=1)
 
        #get q = N(q_mean, q_std)
        q_dist = torch.distributions.Normal(q_mean, torch.exp(q_logvar / 2))

        #get reparameterized sample from N(q_mean, q_std)
        z_rsample = q_dist.rsample()

        #calculate the KL divergence between p and q
        self.loss_kl = q_dist.log_prob(z_rsample) - p_dist.log_prob(z_rsample)

        #project z to the proper dimensions and join with main branch
        tensor = tensor + self.z_projection(z_rsample)

        #pass through the res block
        tensor = self.res(tensor)

        return tensor

    def sample(self, tensor, temp):
        """
        Perform a sampling pass through the convolution module.  The main difference
        between forward() and sample() is which distribution the latent variable z is
        drawn from.  During training z is drawn from the posterior q(z|x), whereas during 
        sampling z is drawn from the prior p(z).
        """      
        
        #get the mean, log-variance of p, and the residual flow to the main path
        theta = self.theta(tensor)
        p_mean, p_logvar, res = torch.tensor_split(theta, (self.z_channels, 2 * self.z_channels), dim=1)

        #modify log-variance of p according to temperature parameter
        p_logvar = p_logvar + torch.full_like(p_logvar, math.log(temp))

        #get p = N(p_mean, p_std)
        p_dist = torch.distributions.Normal(p_mean, torch.exp(p_logvar / 2))

        #get reparameterized sample from N(p_mean, p_std)
        z_rsample = p_dist.rsample()

        #project z to the proper dimensions and join with main branch
        tensor = tensor + self.z_projection(z_rsample)

        #pass through the res block
        tensor = self.res(tensor)

        return tensor


class Block(nn.Sequential):
    """
    This is the basic stacked convolutional block used in other larger blocks.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 res=False,
                 mid_kernel=3):
        super().__init__()

        self.res = res

        padding = 0 if mid_kernel == 1 else 1

        self.append(nn.GELU())
        self.append(nn.Conv2d(in_channels, mid_channels, 1))
        self.append(nn.GELU())
        self.append(nn.Conv2d(mid_channels, mid_channels, mid_kernel, padding=padding))
        self.append(nn.GELU())
        self.append(nn.Conv2d(mid_channels, mid_channels, mid_kernel, padding=padding))
        self.append(nn.GELU())
        self.append(nn.Conv2d(mid_channels, out_channels, 1))

    def forward(self, in_tensor):
        """
        Perform the forward pass through the convolution module.
        """

        out_tensor = super().forward(in_tensor)
        
        #optional residual gradient flow 
        if self.res:
            out_tensor = out_tensor + in_tensor

        return out_tensor


