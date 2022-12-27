#!/usr/bin/python3

"""
Variational Auto Encoder (VAE) model.
"""

import torch
import math
import numpy as np

from torch import nn
from torch.nn import GELU
from torch.distributions.kl import kl_divergence
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from torch.nn.functional import log_softmax
from torch import logsumexp, sigmoid, tanh


class VAE(nn.Module):
    """
    VAE Encoder/Decoder using stride 1 convolutions and avgpool/interpolate for up/down sampling.
    """

    def __init__(self, encoder_layers, decoder_layers):
        super().__init__()

        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers)
        
        #register the training epoch so it is saved with the model's state_dict
        self.register_buffer(torch.tensor([0], dtype=int))

    def encode(self, tensor):
        """
        Pass input tensor through the encoder.
        """
    
        return self.encoder.forward(tensor)

    def decode(self, tensor=None, target=None):
        """
        Pass latent encoding tensor through the decoder.
        """

        return self.decoder.forward(tensor, target=target)

    def sample(self, n_samples, temp=1.0):
        """
        Sample from the model.
        """

        return self.decoder.sample(n_samples, temp)
        #px_z = self.decoder.sample(n_samples, temp)
        #return self.decoder.dmol_net.sample(px_z)

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

        #get the total number of blocks for weight scaling
        n_blocks = sum(layer["n_blocks"] for layer in layers)

        #calculate the weight scaling factor
        final_scale = (1 / n_blocks)**0.5

        self.encoder_groups = nn.Sequential()

        for layer in layers:
            self.encoder_groups.append(EncoderGroup(final_scale=final_scale, **layer))

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
        
        #get the total number of blocks for weight scaling
        n_blocks = sum(layer["n_blocks"] for layer in layers)

        #calculate the weight scaling factor
        final_scale = (1 / n_blocks)**0.5

        for layer in layers:
            self.decoder_groups.append(DecoderGroup(final_scale=final_scale, **layer))

        self.gain = nn.Parameter(torch.ones((1, 512, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, 512, 1, 1)))

        self.out_conv = nn.Conv2d(512, 3, kernel_size=3, padding=1)
        self.mixture_net = MixtureNet(10)

    def forward(self, activations, target):
        """
        Perform a forward pass through the decoder.
        """

        tensor = torch.zeros_like(activations[self.decoder_groups[0].resolution], requires_grad=True)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        for decoder_group in self.decoder_groups:
            tensor = decoder_group(tensor, activations[decoder_group.resolution])

        tensor = tensor * self.gain + self.bias
        
        #return tensor
        
        #return self.out_conv(tensor)

        tensor = self.mixture_net(tensor, target)

        return tensor

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
 
        tensor = tensor * self.gain + self.bias
        
        #return tensor

        #return self.out_conv(tensor)       

        tensor = self.mixture_net.sample(tensor)

        return tensor
 
    def reconstruct(self, activations):
        """
        Perform a forward pass through the decoder.
        """

        tensor = torch.zeros_like(activations[self.decoder_groups[0].resolution], requires_grad=True)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        for decoder_group in self.decoder_groups:
            tensor = decoder_group(tensor, activations[decoder_group.resolution])

        tensor = tensor * self.gain + self.bias
        
        #return tensor
        
        #return self.out_conv(tensor)

        tensor = self.mixture_net.sample(tensor)

        return tensor

       
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
                 downsample_ratio=2,
                 final_scale=1):
        super().__init__()

        self.resolution = resolution
        self.downsample_ratio = downsample_ratio

        #ratio per VDVAE
        mid_channels = channels // 4
        
        self.encoder_blocks = nn.Sequential()

        #use a (3, 3) kernel if the resolution > 2
        mid_kernel = 3 if resolution > 2 else 1
        
        for _ in range(n_blocks):
            self.encoder_blocks.append(Block(channels, mid_channels, channels, mid_kernel=mid_kernel, final_scale=final_scale))
        
       
    def forward(self, tensor):
        """
        Perform the forward pass through the convolution module.  Store the activations.
        """

        tensor = self.encoder_blocks.forward(tensor)
        
        #store the activations
        self.activations = tensor
        
        #downsample
        if self.downsample_ratio > 0:
            tensor = nn.functional.avg_pool2d(tensor, self.downsample_ratio)
        
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
                 upsample_ratio=2,
                 final_scale=1):
        super().__init__()
      
        self.upsample_ratio=upsample_ratio
        self.channels = channels
        self.resolution = resolution
        self.decoder_blocks = nn.ModuleList()

        for _ in range(n_blocks):
            self.decoder_blocks.append(DecoderBlock(channels, resolution, final_scale))
        
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
                 resolution,
                 final_scale):
        super().__init__()

        #mid_channels and z_channels ratios from VDVAE
        mid_channels = channels // 4

        #z_channels fixed at 16
        self.z_channels = 16
     
        #use a (3, 3) kernel if the resolution > 2
        mid_kernel = 3 if resolution > 2 else 1

        #block ratios from VDVAE
        self.phi = Block(channels * 2, mid_channels, 2 * self.z_channels, mid_kernel=mid_kernel)
        self.theta = Block(channels, mid_channels, channels + (2 * self.z_channels), mid_kernel=mid_kernel)
        self.z_projection = nn.Conv2d(self.z_channels, channels, kernel_size=1)
        self.res = Block(channels, mid_channels, channels, res=True, mid_kernel=mid_kernel, final_scale=final_scale)

        self.z_projection.weight.data *= final_scale

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
        p_dist = Normal(p_mean, torch.exp(p_logvar / 2))

        #join the output from theta with the main branch
        tensor = tensor + res

        #get the mean/log-variance of q
        q_mean, q_logvar = self.phi(torch.cat((tensor, activations), dim=1)).chunk(2, dim=1)

        #get q = N(q_mean, q_std)
        q_dist = Normal(q_mean, torch.exp(q_logvar / 2))

        #get reparameterized sample from N(q_mean, q_std)
        z_rsample = q_dist.rsample()

        #calculate the KL divergence between p and q
        #I'm not conviced this is actually correct
        #log_prob() returns the log of PDF(distribution|sample)
        #this will minimize the difference between log_prob, but that could be accomplished by 
        #learning hyperparameters which make z_sample very improbable
        #unless there's something I'm missing here
        #self.loss_kl = (q_dist.log_prob(z_rsample) - p_dist.log_prob(z_rsample)).abs()
        self.loss_kl = kl_divergence(p_dist, q_dist)

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
        p_dist = Normal(p_mean, torch.exp(p_logvar / 2))

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
                 mid_kernel=3,
                 final_scale=1):
        super().__init__()

        self.res = res

        padding = 0 if mid_kernel == 1 else 1

        self.append(GELU())
        self.append(nn.Conv2d(in_channels, mid_channels, 1))
        self.append(GELU())
        self.append(nn.Conv2d(mid_channels, mid_channels, mid_kernel, padding=padding))
        self.append(GELU())
        self.append(nn.Conv2d(mid_channels, mid_channels, mid_kernel, padding=padding))
        self.append(GELU())
        self.append(nn.Conv2d(mid_channels, out_channels, 1))

        self[-1].weight.data *= final_scale
            
    def forward(self, in_tensor):
        """
        Perform the forward pass through the convolution module.
        """

        out_tensor = super().forward(in_tensor)
        
        #optional residual gradient flow 
        if self.res:
            out_tensor = out_tensor + in_tensor

        return out_tensor


class MixtureNet(nn.Module):
    """
    The output of the encoder is fed to the MixtureNet.  Here the probability distribution of a
    pixel's value is modeled by a mixture of distributions parameterized by the decoder output.
    """

    def __init__(self, n_mixtures, link_scaler=2):
        super().__init__()

        self.n_mixtures = n_mixtures
        self.link_scaler = link_scaler #used to scale the output of the link function

        #convs for generating dist parameters for the red sub-pixel
        self.dist_r_0 = nn.Conv2d(512, n_mixtures, 1)
        self.dist_r_1 = nn.Conv2d(512, n_mixtures, 1)
        
        #convs for generating dist parameters for the green sub-pixel
        self.dist_g_0 = nn.Conv2d(512, n_mixtures, 1)
        self.dist_g_1 = nn.Conv2d(512, n_mixtures, 1)
        
        #convs for generating dist parameters for the blue sub-pixel
        self.dist_b_0 = nn.Conv2d(512, n_mixtures, 1)
        self.dist_b_1 = nn.Conv2d(512, n_mixtures, 1)

        #conv for generating the mixture score
        self.mix_score = nn.Conv2d(512, n_mixtures, 1)

    def link(self, tensor):
        #link function between the parameter conv output and the dist distributions
        #input range (-inf, inf), output range (0, link_scaler)
        #self.link = lambda x: self.link_scaler * sigmoid(x)
        #self.link = lambda x: tanh(x)
        #return 1 + (0.2 * sigmoid(tensor))
        return tensor

    def forward(self, dec_out, target):
        """
        Find the negative log likelihood of the target given a mixture of dist distributions
        parameterized by the output of the decoder net.
        """
       
        CLAMP = (-torch.inf, torch.inf)

        dist_r, dist_g, dist_b, log_prob_mix = self.get_distributions(dec_out)
        
        #the Betas generate samples of shape N x n_mixtures x H x W
        #expand/repeat the target tensor along the singleton color channels, maintain 4 dimensions
        #shape N x n_mixtures x H x W
        target_r = target[:, 0:1].expand(-1, self.n_mixtures, -1, -1)
        target_g = target[:, 1:2].expand(-1, self.n_mixtures, -1, -1)
        target_b = target[:, 2:3].expand(-1, self.n_mixtures, -1, -1)

        #get the log probabilities of the target given the dists with current parameters
        #shape N x n_mixtures x H x W
        log_prob_r = dist_r.log_prob(target_r)
        log_prob_g = dist_g.log_prob(target_g)
        log_prob_b = dist_b.log_prob(target_b)
        
        #clamp inf values
        log_prob_r = log_prob_r.clamp(*CLAMP)
        log_prob_g = log_prob_g.clamp(*CLAMP)
        log_prob_b = log_prob_b.clamp(*CLAMP)
        
        #sum the log probabilities of each color channel and modify by the softmax of the  mixture score
        #shape N x n_mixtures x H x W
        log_prob = log_prob_r + log_prob_g + log_prob_b + log_prob_mix

        log_prob = log_prob.clamp(*CLAMP)
        
        #exponentiate, sum, take log along the dimension of each dist
        #shape N x H x W
        log_prob = logsumexp(log_prob, dim=1)

        #clamp inf values
        log_prob = log_prob.clamp(*CLAMP)

        #scale by the number of elements per batch item
        #shape N
        log_prob = log_prob.sum(dim=(1, 2)) / target[0].numel()

        #clamp inf values
        log_prob = log_prob.clamp(*CLAMP)

        #return the negation of the log likelihood suitable for minimization
        return -log_prob

    def sample(self, dec_out):
        """
        Sample from a mixture of dist distributions
        parameterized by the output of the decoder net.
        """
        
        dist_r, dist_g, dist_b, log_prob_mix = self.get_distributions(dec_out)
        
        #choose 1 of n_mixtures distributions based on their log probabilities
        indexes = Categorical(logits=log_prob_mix.permute(0, 2, 3, 1)).sample()

        #get the color channels based on the indexes
        color_r = dist_r.sample()
        color_g = dist_g.sample()
        color_b = dist_b.sample()


        #TODO figure out this indexing
        #this creates a tensor shaped like the color samples:  N x n_mixtures x H x W
        #assigns a value of 1 to the channel corresponding with the selected distribution
        #all others are zero
        indexes2 = torch.zeros_like(color_r)
        for idx in range(color_r.shape[2]):
            for jdx in range(color_r.shape[3]):
                channel = indexes[0, idx, jdx]
                indexes2[0, channel, idx, jdx] = 1.0

        #pointwise multiplies with the color samples and sums along the channels axis
        #now all values are zeroed except those of the selected distributions
        color_r = (color_r * indexes2).sum(dim=1)
        color_g = (color_g * indexes2).sum(dim=1)
        color_b = (color_b * indexes2).sum(dim=1)
        #these are now shaped N x 1 x H x W

        #stack the color channels
        img = torch.cat((color_r, color_g, color_b), dim=0).unsqueeze(0)
        #shape N x 3 x H x W

        return img

    def get_distributions(self, dec_out):
        """
        """

        #dist parameters for the red sub-pixel distributions
        #shape N x n_mixtures x H x W
        dist_r_0 = self.link(self.dist_r_0(dec_out))
        dist_r_1 = self.link(self.dist_r_1(dec_out))
        
        #dist parameters for the blue sub-pixel distributions
        #shape N x n_mixtures x H x W
        dist_g_0 = self.link(self.dist_g_0(dec_out))
        dist_g_1 = self.link(self.dist_g_1(dec_out))
 
        #dist parameters for the green sub-pixel distributions
        #shape N x n_mixtures x H x W
        dist_b_0 = self.link(self.dist_b_0(dec_out))
        dist_b_1 = self.link(self.dist_b_1(dec_out))
 
        #log probability of each mixture for each distribution
        #shape N x n_mixtures x H x W
        log_prob_mix = log_softmax(self.mix_score(dec_out), dim=1)

        #initialize the distributions
        #shape N x n_mixtures x H x W
        #dist_r = Beta(dist_r_0, dist_r_1, validate_args=True)
        #dist_g = Beta(dist_g_0, dist_g_1, validate_args=True)
        #dist_b = Beta(dist_b_0, dist_b_1, validate_args=True)

        dist_r = Normal(dist_r_0, torch.exp(dist_r_1 / 2), validate_args=True)
        dist_g = Normal(dist_g_0, torch.exp(dist_g_1 / 2), validate_args=True)
        dist_b = Normal(dist_b_0, torch.exp(dist_b_1 / 2), validate_args=True)

        return dist_r, dist_g, dist_b, log_prob_mix


class GELU6(nn.GELU):
    """
    This non-linearity combines the GELU profile with a clamp function to constrain the
    output to 6 or less, similar to the ReLU6 non-linearity.
    """

    def forward(self, tensor):
        return super().forward(tensor).clamp(max=6)
