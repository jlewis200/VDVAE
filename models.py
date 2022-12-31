#!/usr/bin/python3

"""
Variational Auto Encoder (VAE) model.
"""

import torch

from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.nn.functional import interpolate, log_softmax
from torch import logsumexp, sigmoid, tanh
from time import time

from vae_helpers import DmolNet


class VAE(nn.Module):
    """
    VAE Encoder/Decoder using stride 1 convolutions and avgpool/interpolate for down/up sampling.
    """

    def __init__(self, encoder_layers, decoder_layers, low_bit=False):
        super().__init__()
        
        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers, low_bit=low_bit)
        
        #register the training epoch and start time so it is saved with the model's state_dict
        self.register_buffer("epoch", torch.tensor([0], dtype=int))

        #the model is identified by the initial creation time
        self.register_buffer("start_time", torch.tensor([int(time())], dtype=int))

        #function pointers to corresponding encoder/decoder functions
        self.decode = self.decoder.forward
        self.sample = self.decoder.sample
        self.get_loss_kl = self.decoder.get_loss_kl

    @property
    def device(self):
        """
        Return the pytorch device where input tensors should be located.
        """

        return self.encoder.device

    def get_nll(self, tensor, target):
        """
        Apply the loss target transform and pass to the decoder.
        """
 
        #TODO:  find a better way to do this
        #the ffhq dataset is stored in [0, 255]
        if target.dtype == torch.uint8:
            target = target.to(torch.float32) / 255
            target = target.permute(0, 3, 1, 2)

   
        target = self.transform_target(target)

        return self.decoder.get_nll(tensor, target)

    def encode(self, tensor):
        """
        Apply the model input transform and pass to encoder.
        """

        #TODO:  find a better way to do this
        #the ffhq dataset is stored in [0, 255]
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255
            tensor = tensor.permute(0, 3, 1, 2)

        tensor = self.transform_in(tensor)

        return self.encoder.forward(tensor)

    def reconstruct(self, tensor):
        """
        Reconstruct an image.
        """

        px_z = self.decode(self.encode(tensor))
        
        return self.decoder.out_net.sample(px_z)

class Encoder(nn.Module):
    """
    VAE Encoder class.
    """
    
    def __init__(self, layers):
        super().__init__()

        self.in_conv = nn.Conv2d(3, layers[0]["channels"], kernel_size=3, padding=1)

        #get the total number of blocks for weight scaling
        n_blocks = sum(layer["n_blocks"] for layer in layers)

        #calculate the weight scaling factor
        final_scale = (1 / n_blocks)**0.5

        #construct the encoder groups
        self.encoder_groups = nn.Sequential()

        for layer in layers:
            self.encoder_groups.append(EncoderGroup(final_scale=final_scale, **layer))

    @property
    def device(self):
        """
        Return the pytorch device where input tensors should be located.
        """

        return self.in_conv.weight.device

    def forward(self, tensor):
        """
        Perform a forward pass through the encoder.
        """

        #input convolution
        tensor = self.in_conv(tensor)

        #encode the input
        self.encoder_groups(tensor)

        #collect the activations from the encoding process
        activations = {}

        for encoder_group in self.encoder_groups:
            activations[encoder_group.resolution] = encoder_group.activations

        return activations

class Decoder(nn.Module):
    """
    VAE Decoder class.
    """

    def __init__(self, layers, low_bit=False):
        super().__init__()
        
        #get the total number of blocks for weight scaling
        n_blocks = sum(layer["n_blocks"] for layer in layers)

        #calculate the weight scaling factor
        final_scale = (1 / n_blocks)**0.5

        #construct the decoder_groups
        self.decoder_groups = nn.ModuleList()
        
        for layer in layers:
            self.decoder_groups.append(DecoderGroup(final_scale=final_scale, **layer))

        channels = layers[-1]["channels"]
        self.gain = nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))

        #self.out_conv = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        #self.mixture_net = MixtureNet(10, channels)
        #self.out_net = DmolNet(channels, 10, low_bit)
        self.dmll_net = DmllNet(channels, 10, 8)

    def forward(self, activations=None, temp=0):
        """
        Perform a forward pass through the decoder.
        """

        if activations is not None:
            #non-None activations indicates a training pass
            tensor = torch.zeros_like(activations[self.decoder_groups[0].resolution], requires_grad=True)

        else:
            #None activations indicates a sampling pass
            tensor = torch.zeros_like(self.decoder_groups[0].bias, requires_grad=True)

        tensor = tensor.to(self.gain.device)

        #pass through all decoder groups
        for decoder_group in self.decoder_groups:
            group_activations = activations[decoder_group.resolution] if activations is not None else None
            tensor = decoder_group(tensor, activations=group_activations, temp=temp)

        #apply gain/bias
        return tensor * self.gain + self.bias

    def sample(self, n_samples, temp=0):
        """
        Sample from the model.  Temperature scales the standard deviation of the distributions the
        latent encodings are drawn from.  A temperature of 1 corresponds with the original standard
        deviation.
        """
        
        #collect the tensor encoding of the posterior
        px_z = self.forward(activations=None, temp=temp)
   
        #sample from the reconstruction network
        return self.out_net.sample(px_z)

    def reconstruct(self, activations, temp=0):
        """
        Perform a forward pass through the decoder.
        """

        #collect the tensor encoding of the input
        tensor = self.forward(activations=activations)
        
        #sample from the reconstruction network
        return self.out_net.sample(tensor, temp=temp)
 
    def get_nll(self, tensor, target):
        """
        Get the negative log likelihood of the tensor given target tensor.
        """

        #return self.mixture_net(tensor, target)      
        #return self.out_net.nll(tensor, target)      
        return self.dmll_net.get_nll(tensor, target)      

    def get_loss_kl(self):
        """
        Collect the KL divergence loss from all groups.
        """

        #gather and sum the KL losses from each decoder group (gradients are preserved)
        loss_kl = sum(group.get_loss_kl() for group in self.decoder_groups)

        #scale by the original resolution and color depth
        return loss_kl / (3 * self.decoder_groups[-1].resolution**2)


class EncoderGroup(nn.Module):
    """
    Encoder group encapsulates all encoder blocks at the same tensor H x W.
    """
    
    def __init__(self, channels, resolution, n_blocks, final_scale=1):
        super().__init__()

        self.resolution = resolution

        #ratio per VDVAE
        mid_channels = channels // 4
        
        self.encoder_blocks = nn.Sequential()

        #use a (3, 3) kernel if the resolution > 2
        mid_kernel = 3 if resolution > 2 else 1

        for _ in range(n_blocks):
            self.encoder_blocks.append(Block(channels, mid_channels, channels, mid_kernel=mid_kernel, final_scale=final_scale, res=True))
        
    def forward(self, tensor):
        """
        Perform the forward pass through the convolution module.  Store the activations.
        """

        #downsample if necessary
        if tensor.shape[-1] != self.resolution:
            kernel_size = tensor.shape[-1] // self.resolution
            tensor = nn.functional.avg_pool2d(tensor, kernel_size)

        #the original VDVAE uses the activations of the 2nd to last encoder block at each H/W
        #except the final 1x1, then the final activation is stored
        #to make use of the pretrained weights we do the same here
        for encoder_block in self.encoder_blocks[:-1]:
            tensor = encoder_block.forward(tensor)
        
        #store the activations
        self.activations = tensor

        #pass through the final decoder block
        tensor = self.encoder_blocks[-1].forward(tensor)

        #store the final activations if this is the N x C x 1 x 1 encoder group
        if self.resolution == 1:
            self.activations = tensor

        return tensor


class DecoderGroup(nn.Module):
    """
    Decoder group encapsulates all decder blocks at the same tensor H x W.
    """

    def __init__(self, channels, n_blocks, resolution, bias=True, final_scale=1):
        super().__init__()
      
        self.channels = channels
        self.resolution = resolution

        #construct the decoder blocks
        self.decoder_blocks = nn.ModuleList()

        for _ in range(n_blocks):
            self.decoder_blocks.append(DecoderBlock(channels, resolution, final_scale))
        
        self.bias = None
       
        #register the bias as a learnable parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, channels, resolution, resolution)))

    def forward(self, tensor, activations=None, temp=0):
        """
        Perform the forward pass through this group.
        """

        #the input tensor will be None for the first decoder group
        if tensor is None:
            tensor = torch.zeros_like(self.bias, requires_grad=True)
        
        #upsample the output of the previous decoder group to match the reolution of this decoder group
        tensor = interpolate(tensor, self.resolution)
      
        if self.bias is not None:
            #add the bias term for this decoder group
            tensor = tensor + self.bias
         
        #pass through the decoder blocks
        for decoder_block in self.decoder_blocks:
            tensor = decoder_block(tensor, activations=activations, temp=temp)

        return tensor


    def get_loss_kl(self):
        """
        Get the KL divergence loss for each decoder block in this decoder group.
        """

        #gather and sum the KL losses from each decder block (gradients are preserved)
        return sum(block.loss_kl.sum(dim=(1, 2, 3)) for block in self.decoder_blocks)


class DecoderBlock(nn.Module):
    """
    Decoder block per VDVAE architecture.
    """

    def __init__(self, channels, resolution, final_scale):
        super().__init__()

        self.channels = channels

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

    def forward(self, tensor, activations=None, temp=0):
        """
        Perform a forward pass through the convolution module.  This method is used when training
        and when sampling from the model.  The main difference is which distribution the latent
        variable z is drawn from.  During training z is drawn from the posterior q(z|x), whereas
        during sampling z is drawn from the prior p(z).
        """      

        if activations is None:
            return self.sample(tensor, activations, temp)
 
        #get the mean/log-standard-deviation of q/p, and the residual flow to the main path
        q_mean, q_logstd = self.phi(torch.cat((tensor, activations), dim=1)).split(self.z_channels, dim=1)
        p_mean, p_logstd, res = self.theta(tensor).split((self.z_channels, self.z_channels, self.channels), dim=1)

        #get q = N(q_mean, q_std), p = N(p_mean, p_std), and KL divergence from q to p
        q_dist = Normal(q_mean, torch.exp(q_logstd))
        p_dist = Normal(p_mean, torch.exp(p_logstd))
        self.loss_kl = kl_divergence(q_dist, p_dist)
 
        #join the output from theta with the main branch
        tensor = tensor + res

        #get reparameterized sample from N(q_mean, q_std)
        z_rsample = q_dist.rsample()

        #project z to the proper dimensions and join with main branch
        tensor = tensor + self.z_projection(z_rsample)

        #pass through the res block
        return self.res(tensor)

    def sample(self, tensor, activations=None, temp=0):
        """
        Perform a sampling pass through the convolution module.
        """      

        #get the mean, log-standard-deviation of p, and the residual flow to the main path
        p_mean, p_logstd, res = self.theta(tensor).split((self.z_channels, self.z_channels, self.channels), dim=1)

        #get p = N(p_mean, p_std)
        p_dist = Normal(p_mean, temp * torch.exp(p_logstd))

        #join the output from theta with the main branch
        tensor = tensor + res

        #get reparameterized sample from N(p_mean, p_std)
        z_rsample = p_dist.rsample()
        
        #project z to the proper dimensions and join with main branch
        tensor = tensor + self.z_projection(z_rsample)

        #pass through the res block
        return self.res(tensor)


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

        self.append(nn.GELU())
        self.append(nn.Conv2d(in_channels, mid_channels, 1))
        self.append(nn.GELU())
        self.append(nn.Conv2d(mid_channels, mid_channels, mid_kernel, padding=padding))
        self.append(nn.GELU())
        self.append(nn.Conv2d(mid_channels, mid_channels, mid_kernel, padding=padding))
        self.append(nn.GELU())
        self.append(nn.Conv2d(mid_channels, out_channels, 1))

        #scale the final convolution initial weights by final_scale
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

    def __init__(self, n_mixtures, channels, link_scaler=2):
        super().__init__()

        self.n_mixtures = n_mixtures
        self.link_scaler = link_scaler #used to scale the output of the link function

        #convs for generating dist parameters for the red sub-pixel
        self.dist_r_0 = nn.Conv2d(channels, n_mixtures, 1)
        self.dist_r_1 = nn.Conv2d(channels, n_mixtures, 1)
        
        #convs for generating dist parameters for the green sub-pixel
        self.dist_g_0 = nn.Conv2d(channels, n_mixtures, 1)
        self.dist_g_1 = nn.Conv2d(channels, n_mixtures, 1)
        
        #convs for generating dist parameters for the blue sub-pixel
        self.dist_b_0 = nn.Conv2d(channels, n_mixtures, 1)
        self.dist_b_1 = nn.Conv2d(channels, n_mixtures, 1)

        #conv for generating the mixture score
        self.mix_score = nn.Conv2d(channels, n_mixtures, 1)

    def link(self, tensor):
        #link function between the parameter conv output and the dist distributions
        #input range (-inf, inf), output range (0, link_scaler)
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
        #dist_r = Beta(dist_r_0, dist_r_1)
        #dist_g = Beta(dist_g_0, dist_g_1)
        #dist_b = Beta(dist_b_0, dist_b_1)

        dist_r = Normal(dist_r_0, torch.exp(dist_r_1))
        dist_g = Normal(dist_g_0, torch.exp(dist_g_1))
        dist_b = Normal(dist_b_0, torch.exp(dist_b_1))

        return dist_r, dist_g, dist_b, log_prob_mix


class DmllNet(nn.Module):
    """
    """

    def __init__(self, channels, n_mixtures, bits=8):
        super().__init__()

        self.n_mixtures = n_mixtures
        self.bits = bits

        #convs for generating red sub-pixel distribution parameters 
        self.r_mean = nn.Conv2d(channels, n_mixtures, 1)
        self.r_logscale = nn.Conv2d(channels, n_mixtures, 1)
        
        #convs for generating green sub-pixel distribution parameters
        self.g_mean = nn.Conv2d(channels, n_mixtures, 1)
        self.g_logscale = nn.Conv2d(channels, n_mixtures, 1)
        self.gr_coeff = nn.Conv2d(channels, n_mixtures, 1)
        
        #convs for generating blue sub-pixel distribution parameters
        self.b_mean = nn.Conv2d(channels, n_mixtures, 1)
        self.b_logscale = nn.Conv2d(channels, n_mixtures, 1)
        self.br_coeff = nn.Conv2d(channels, n_mixtures, 1)
        self.bg_coeff = nn.Conv2d(channels, n_mixtures, 1)

        #conv for generating the log-probabilities of each of the n mixtures
        self.logits = nn.Conv2d(channels, n_mixtures, 1)

    def get_nll(self, dec_out, target):
        """
        Find the negative log likelihood of the target given a mixture of dist distributions
        parameterized by the output of the decoder net.
        """

        target = target.to(dec_out.device)
        
        #the Betas generate samples of shape N x n_mixtures x H x W
        #expand/repeat the target tensor along the singleton color channels, maintain 4 dimensions
        #shape N x n_mixtures x H x W
        target_r = target[:, 0:1].expand(-1, self.n_mixtures, -1, -1)
        target_g = target[:, 1:2].expand(-1, self.n_mixtures, -1, -1)
        target_b = target[:, 2:3].expand(-1, self.n_mixtures, -1, -1)

        dist_r, dist_g, dist_b, log_prob_mix = self.get_distributions(dec_out, target_r, target_g, target_b)
        
        #get the log probabilities of the target given the dists with current parameters
        #shape N x n_mixtures x H x W
        log_prob_r = dist_r.log_prob(target_r)
        log_prob_g = dist_g.log_prob(target_g)
        log_prob_b = dist_b.log_prob(target_b)
        
        #sum the log probabilities of each color channel and modify by the softmax of the  mixture score
        #shape N x n_mixtures x H x W
        log_prob = log_prob_r + log_prob_g + log_prob_b + log_softmax(log_prob_mix, dim=1)
        
        #exponentiate, sum, take log along the dimension of each dist
        #shape N x H x W
        log_prob = logsumexp(log_prob, dim=1)

        #scale by the number of elements per batch item
        #shape N
        nll = -log_prob.sum(dim=(1, 2)) / target[0].numel()

        #return the negative log likelihood suitable for minimization
        return nll

    def get_distributions(self, dec_out, target_r, target_g, target_b):
        """
        """
        
        #dist parameters for the red sub-pixel distributions
        #shape N x n_mixtures x H x W
        r_mean = self.r_mean(dec_out)
        r_logscale = self.r_logscale(dec_out).clamp(min=-7)
        
        #dist parameters for the blue sub-pixel distributions
        #shape N x n_mixtures x H x W
        g_mean = self.g_mean(dec_out)
        g_logscale = self.g_logscale(dec_out).clamp(min=-7)
        gr_coeff = torch.tanh(self.gr_coeff(dec_out))
 
        #dist parameters for the green sub-pixel distributions
        #shape N x n_mixtures x H x W
        b_mean = self.b_mean(dec_out)
        b_logscale = self.b_logscale(dec_out).clamp(min=-7)
        br_coeff = torch.tanh(self.br_coeff(dec_out))
        bg_coeff = torch.tanh(self.bg_coeff(dec_out))
        
        #log probability of each mixture for each distribution
        #shape N x n_mixtures x H x W
        logits = self.logits(dec_out)

        #mix the mean of green sub-pixel with red
        #this relates the green sub-pixel to the red sub-pixel for conditional sampling
        g_mean = g_mean + (gr_coeff * target_r)

        #mix the mean of blue sub-pixel with red/green
        #this relates the blue sub-pixel to the red/green sub-pixels for conditional sampling
        b_mean = b_mean + (br_coeff * target_r) + (bg_coeff * target_g)

        #initialize the distributions
        dlog_r = DiscreteLogistic(r_mean, torch.exp(r_logscale), self.bits)
        dlog_g = DiscreteLogistic(g_mean, torch.exp(g_logscale), self.bits)
        dlog_b = DiscreteLogistic(b_mean, torch.exp(b_logscale), self.bits)

        return dlog_r, dlog_g, dlog_b, logits


class DiscreteLogistic(TransformedDistribution):
    """
    """

    def __init__(self, mean, scale, bits=8):
        self.bits = bits
        self.half_width = (1 / ((2**bits) - 1))
        self._mean = mean
        
        base_distribution = Uniform(torch.zeros_like(mean), torch.ones_like(mean))
        transforms = [SigmoidTransform().inv, AffineTransform(loc=mean, scale=scale)]
        super().__init__(base_distribution, transforms)
        

    def log_prob(self, value):
        """
        The DiscreteLogistic distribution is built from a TransformedDistribution to create a
        continuous logistic distribution.  This log_prob function uses the TransformedDistribution's
        cdf() and log_prob() functions to return the discretized log probability of value.
        """

        #use a numerical stability term to prevent log(0)
        stability = 1e-12
        prob_threshold = 1e-5

        #use the non-discrete log-probability as a base
        #this value is used when prob < prob_threshold (indicating the distribution parameters are off by quite a bit)
        log_prob = super().log_prob(value)
 
        #find the discrete non-log probability of laying within the bucket
        prob = (self.cdf(value + self.half_width) - self.cdf(value - self.half_width)).clamp(min=stability)
        
        #if the non-log probability is above a threshold, 
        #replace continuous log-probability with the discrete log-probability
        mask = prob > prob_threshold
        log_prob[mask] = prob[mask].log()

        #edge case at 0:  replace 0 with cdf(0 + half_half_width)
        mask = value < -0.999
        log_prob[mask] = self.cdf(value + self.half_width)[mask].log()

        #edge case at 1:  replace 1 with (1 - cdf(1 - half bucket list))
        mask = value > 0.999
        log_prob[mask] = (1 - self.cdf(value - self.half_width))[mask].log()

        return log_prob

