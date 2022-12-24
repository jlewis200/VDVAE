#!/usr/bin/python3

"""
Variational Auto Encoder (VAE) encoder/decoder models.
"""

import torch
import math
import numpy as np

from torch import nn
from torch.nn import GELU, Conv2d
from torch.nn.functional import interpolate, avg_pool2d
from code import interact


class View(nn.Module):
    """
    Pytorch module of view change.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, tensor):
        """
        Return a view of the tensor in the correct shape.
        """

        return tensor.view((-1, *self.shape))


class EncoderMetaBlock(nn.Module):
    """
    Encoder meta block encapsulates all encoder blocks at the same tensor H x W.
    """
    
    def __init__(self,
                 channels,
                 resolution,
                 n_encoder_blocks,
                 n_downsamples=1):

        super().__init__()

        self.n_downsamples = n_downsamples

        #ratio per VDVAE
        mid_channels = channels // 4
        
        self.encoder_blocks = nn.Sequential()

        #use a (3, 3) kernel if the resolution > 1
        mid_kernel = 3 if resolution > 1 else 1
        
        for _ in range(n_encoder_blocks):
            self.encoder_blocks.append(Block(channels, mid_channels, channels, mid_kernel=mid_kernel))
        
        #self.encoder_blocks[-1].weight.data *= np.sqrt(1 / n_encoder_blocks)
       
    def forward(self, tensor):
        """
        Perform the forward pass through the convolution module.  Store the
        output out_tensor.
        """

        tensor = self.encoder_blocks.forward(tensor)
        
        #store the activations
        self.activations = tensor
        
        #downsample
        for _ in range(self.n_downsamples):
            tensor = avg_pool2d(tensor, 2)
        
        return tensor


class Block(nn.Sequential):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 res=False,
                 mid_kernel=3):

        super().__init__()

        self.res = res

        padding = 0 if mid_kernel == 1 else 1

        self.append(GELU())
        self.append(Conv2d(in_channels, mid_channels, 1))
        self.append(GELU())
        self.append(Conv2d(mid_channels, mid_channels, mid_kernel, padding=padding))
        self.append(GELU())
        self.append(Conv2d(mid_channels, mid_channels, mid_kernel, padding=padding))
        self.append(GELU())
        self.append(Conv2d(mid_channels, out_channels, 1))

    def forward(self, in_tensor):
        """
        Perform the forward pass through the convolution module.  Store the
        output out_tensor.
        """

        out_tensor = super().forward(in_tensor)
        
        #optional residual gradient flow 
        if self.res:
            out_tensor = out_tensor + in_tensor

        return out_tensor


class DecoderMetaBlock(nn.Module):
    """
    Decoder meta block encapsulates all decder blocks at the same tensor H x W.
    """

    def __init__(self,
                 channels,
                 n_decoder_blocks,
                 resolution,
                 bias=True,
                 upsample_ratio=2,
                 encoder_meta_block=None):
        
        super().__init__()
      
        self.upsample_ratio=upsample_ratio
        self.channels = channels
        self.resolution = resolution
        self.encoder_meta_block=encoder_meta_block
        self.decode_blocks = nn.Sequential()

        for _ in range(n_decoder_blocks):
            self.decode_blocks.append(DecoderBlock(channels, resolution, encoder_meta_block=encoder_meta_block))
        
        self.activations = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, channels, resolution, resolution)))

    def forward(self, tensor=None):  #output of previous decoder-meta block, or None for first decoder-meta
        """
        Perform the forward pass through the convolution module.  Store the
        output out_tensor.
        """

        if tensor is None:
            #encoder information only enters the decoding path through reparameterized sample of z
            #tensor will be None for the first decoder meta-block
            tensor = torch.zeros((1, self.channels, self.resolution, self.resolution))

            if torch.cuda.is_available():
                tensor = tensor.cuda()

        else:
            #upsample the output of the previous decoder meta block
            tensor = interpolate(tensor, scale_factor=self.upsample_ratio)
      
        if self.bias is not None:
            #add the bias term for this decoder-meta block
            tensor = tensor + self.bias
           
        return self.decode_blocks(tensor)

    def sample(self, tensor=None, temp=1.0):  #output of previous decoder-meta block, or None for first decoder-meta
        """
        """

        if tensor is None:
            #encoder information only enters the decoding path through reparameterized sample of z
            #tensor will be None for the first decoder meta-block
            tensor = torch.zeros((1, channels, self.resolution, self.resolution))

            if torch.cuda.is_available():
                tensor = tensor.cuda()

        else:
            #upsample the output of the previous decoder meta block
            tensor = interpolate(tensor, scale_factor=self.upsample_ratio)
      
        if self.bias:
            #add the bias term for this decoder-meta block
            tensor = tensor + self.bias
        
        for decode_block in self.decode_blocks:
            tensor = decode_block.sample(tensor, temp)

        return tensor

    def get_loss_kl(self):
        """
        Get the KL divergence loss for each decoder block in this decoder meta block.
        """
       
        #begin with a zero-tensor
        loss_kl = torch.zeros(self.decode_blocks[0].loss_kl.shape[0], requires_grad=True)

        if torch.cuda.is_available():
            loss_kl = loss_kl.cuda()

        #sum the KL losses from each decder block
        for decode_block in self.decode_blocks:
            loss_kl = loss_kl + decode_block.loss_kl.sum(dim=(1, 2, 3))

        return loss_kl


class DecoderBlock(nn.Module):
    """
    Decoder block per VDVAE architecture.
    """

    def __init__(self,
                 channels,
                 resolution,
                 encoder_meta_block=None):
        
        super().__init__()

        #store a reference to the encoder_meta_block at this tensor resolution
        self.encoder_meta_block = encoder_meta_block

        #mid_channels and z_channels ratios from VDVAE
        mid_channels = channels // 4

        #z_channels fixed at 16
        self.z_channels = 16
     
        #use a (3, 3) kernel if the resolution > 1
        mid_kernel = 3 if resolution > 1 else 1

        #block ratios from VDVAE
        self.phi = Block(channels * 2, mid_channels, 2 * self.z_channels, mid_kernel=mid_kernel)
        self.theta = Block(channels, mid_channels, channels + (2 * self.z_channels), mid_kernel=mid_kernel)
        self.z_projection = Conv2d(self.z_channels, channels, kernel_size=1)
        self.res = Block(channels, mid_channels, channels, res=True, mid_kernel=mid_kernel)

    def forward(self, tensor):
        """
        Perform a forward pass through the convolution module.

        Note:  A ValueError during the creation of either normal distribution is a sign the 
        learning rate is too high.  The mins/maxs of p_logvar have been observed to trend toward
        inf/-inf when the LR is too high and training collapses.  The magnitude of the mean/logvar
        tend to increase as the tensor resolution increases.
        """      
       
        try:
            #get the mean, log-variance of p, and the residual flow to the main path
            theta = self.theta(tensor)
            p_mean, p_logvar, res = torch.tensor_split(theta, (self.z_channels, 2 * self.z_channels), dim=1)

            #get p = N(p_mean, p_std)
            p_dist = torch.distributions.Normal(p_mean, torch.exp(p_logvar / 2))

            #join the output from theta with the main branch
            tensor = tensor + res

            #get the activations from the paired-encoder block
            enc_activations = self.encoder_meta_block.activations
            
            #get the mean/log-variance of q
            q_mean, q_logvar = self.phi(torch.cat((tensor, enc_activations), dim=1)).chunk(2, dim=1)
     
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

        except ValueError:
            print("ValueError:  check p/q distribution parameters.")
            breakpoint()

        return tensor

    def sample(self, tensor, temp):
        """
        Perform a forward pass through the convolution module.  The main difference
        between forward() and sample() is which distribution the latent variable z is
        drawn from.  During training z is drawn from the posterior q(z|x), whereas during 
        sampling z is drawn from the prior p(z).
        """      
        
        #get the mean, log-variance of p, and the residual flow to the main path
        theta = self.theta(tensor)
        p_mean, p_logvar, res = torch.tensor_split(theta, (self.z_channels, 2 * self.z_channels), dim=1)

        #modify log-variance of p according to temperature parameter
        #p_logvar = p_logvar + torch.full_like(p_logvar, math.log(temp))
        p_logvar = p_logvar + torch.ones_like(p_logvar) * np.log(temp)

        #get p = N(p_mean, p_std)
        p_dist = torch.distributions.Normal(p_mean, torch.exp(p_logvar / 2))

        #get reparameterized sample from N(p_mean, p_std)
        z_rsample = p_dist.rsample()

        #project z to the proper dimensions and join with main branch
        tensor = tensor + self.z_projection(z_rsample)

        #pass through the res block
        tensor = self.res(tensor)

        return tensor

class VAE(nn.Module):
    """
    VAE Encoder/Decoder using stride 1 convolutions and avgpool/interpolate for up/down sampling.
    """

    def __init__(self):
        super().__init__()

        ######################################################################## 
        #encoder

        encoder_layers = [
            {"channels": 512, "n_encoder_blocks":  3, "resolution": 128},
            {"channels": 512, "n_encoder_blocks":  8, "resolution":  64},
            {"channels": 512, "n_encoder_blocks": 12, "resolution":  32},
            {"channels": 512, "n_encoder_blocks": 17, "resolution":  16},
            {"channels": 512, "n_encoder_blocks":  7, "resolution":   8},
            {"channels": 512, "n_encoder_blocks":  5, "resolution":   4, "n_downsamples": 2},
            {"channels": 512, "n_encoder_blocks":  4, "resolution":   1, "n_downsamples": 0}]

        #TODO: refactor to remove the input convolution from the encoder blocks sequential
        self.encoder = nn.Sequential()
        self.encoder.append(Conv2d(3, 512, kernel_size=3, padding=1))

        for encoder_layer in encoder_layers:
            self.encoder.append(EncoderMetaBlock(**encoder_layer))
        
        ######################################################################## 
        #decoder

        decoder_layers = [
            {"channels": 512, "n_decoder_blocks":  2, "resolution":   1, "upsample_ratio": 0},
            {"channels": 512, "n_decoder_blocks":  3, "resolution":   4, "upsample_ratio": 4},
            {"channels": 512, "n_decoder_blocks":  4, "resolution":   8},
            {"channels": 512, "n_decoder_blocks":  9, "resolution":  16},
            {"channels": 512, "n_decoder_blocks": 21, "resolution":  32},
            {"channels": 512, "n_decoder_blocks": 13, "resolution":  64},
            {"channels": 512, "n_decoder_blocks":  7, "resolution": 128}]

        self.decoder = nn.Sequential()
        
        for decoder_layer, encoder_meta_block in zip(decoder_layers, self.encoder[::-1]):
            self.decoder.append(DecoderMetaBlock(**decoder_layer, encoder_meta_block=encoder_meta_block))

        #TODO: refactor to remove the output convolution from the encoder blocks sequential
        self.decoder.append(Conv2d(512, 3, kernel_size=3, padding=1))

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

    def get_loss_kl(self):
        #TODO: do this better, this should be shaped like the number of batches
        loss_kl = torch.zeros(self.decoder[0].decode_blocks[0].loss_kl.shape[0],
                              requires_grad=True)

        if torch.cuda.is_available():
            loss_kl = loss_kl.cuda()

        for decode_meta_block in self.decoder:
            if isinstance(decode_meta_block, DecoderMetaBlock):
                #TODO:  fix the scaling, right now theres an output convolution in the decoder as well 
                #TODO:  when that conv is removed, there won't be any need for the isinstance check
                #scale by number of decoder-meta blocks
                loss_kl = loss_kl + decode_meta_block.get_loss_kl()

            #TODO:  do this better, this should be color channels * H * W: 3 * 128 * 128
            scale_factor = 3 * self.decoder[-2].resolution**2
            #TODO:  remove the assert after I fix the abomination above
            assert(scale_factor == 3 * 128 * 128)

            loss_kl = loss_kl / scale_factor

        return loss_kl

    def sample(self, n_samples, temp=1.0):
        """
        Sample from the model.  Temperature controls the variance of the distributions the latent
        encodings are drawn from.  A higher temperature leads to higher variance and more dynamic
        results.
        """

        #set dummy activations in the encoder
        self.encoder[-1].activations = torch.zeros_like(self.decoder[0].bias)
    
        tensor = None

        #TODO:  this slice will have to be removed when the encoder metas get their own sequential
        for decoder_meta_block in self.decoder[:-1]:
            tensor = decoder_meta_block.sample(tensor, temp=temp)
        
        tensor = self.decoder[-1].forward(tensor)

        return tensor
