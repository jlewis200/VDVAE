#!/usr/bin/python3

"""
Very Deep Variational Auto Encoder (VDVAE).
"""

# ignore these pylint findings due to torch lazy loading false positives
# pylint: disable=no-member, no-name-in-module

from time import time

import torch

from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.nn.functional import interpolate, log_softmax, one_hot
from torch import logsumexp, tanh


class VDVAE(nn.Module):
    """
    Very Deep Variational Auto Encoder (VDVAE).
    """

    def __init__(self, encoder_layers, decoder_layers, bits=8):
        super().__init__()

        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers, bits=bits)

        #register the training epoch and start time so it is saved with the model's state_dict
        self.register_buffer("epoch", torch.tensor([0], dtype=int))

        #the model is identified by the initial creation time
        self.register_buffer("start_time", torch.tensor([int(time())], dtype=int))

        #function pointers to corresponding encoder/decoder functions
        self.decode = self.decoder.forward
        self.get_loss_kl = self.decoder.get_loss_kl

    def forward(self, tensor):
        """
        Perform the forward training pass.
        """

        return self.decode(self.encode(tensor))

    def get_nll(self, tensor, target):
        """
        Apply the loss target transform and pass to the decoder.
        """

        target = self.transform_target(target)

        return self.decoder.get_nll(tensor, target)

    def encode(self, tensor):
        """
        Apply the model input transform and pass to encoder.
        """

        tensor = self.transform_in(tensor)

        return self.encoder.forward(tensor)

    def reconstruct(self, tensor):
        """
        Reconstruct an image.
        """

        #get the posterior
        px_z = self.decode(self.encode(tensor))

        #sample from the reconstruction network
        sample = self.decoder.dmll_net.sample(px_z)

        #apply the output transformation
        return self.transform_out(sample)

    def sample(self, n_samples, temp=1.0):
        """
        Sample from the model.
        """

        #sample from the decoder
        sample = self.decoder.sample(n_samples, temp)

        #apply the output transformation
        return self.transform_out(sample)

    @property
    def device(self):
        """
        Return the pytorch device where input tensors should be located.
        """

        return self.encoder.device

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

    def __init__(self, layers, n_mixtures=10, bits=8):
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

        self.dmll_net = DmllNet(channels, n_mixtures, bits=bits)

    def forward(self, activations=None, temp=0):
        """
        Perform a forward pass through the decoder.
        """

        if activations is not None:
            #non-None activations indicates a training pass
            tensor = torch.zeros_like(activations[self.decoder_groups[0].resolution])

        else:
            #None activations indicates a sampling pass
            tensor = torch.zeros_like(self.decoder_groups[0].bias)

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
        return self.dmll_net.sample(px_z)

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
        self.activations = None

        #ratio per VDVAE
        mid_channels = channels // 4

        self.encoder_blocks = nn.Sequential()

        #use a (3, 3) kernel if the resolution > 2
        mid_kernel = 3 if resolution > 2 else 1

        for _ in range(n_blocks):
            self.encoder_blocks.append(Block(channels,
                                             mid_channels,
                                             channels,
                                             mid_kernel=mid_kernel,
                                             final_scale=final_scale,
                                             res=True))

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
        #to make use of the pretrained weights the same is done here
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
            tensor = torch.zeros_like(self.bias)

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
        self.loss_kl = None

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

        #scale the z projection weights as per VDVAE
        self.z_projection.weight.data *= final_scale

    def forward(self, tensor, activations=None, temp=0):
        """
        Perform a forward pass through the convolution module.  This method is used when training
        and when sampling from the model.  The main difference is which distribution the latent
        variable z is drawn from.  During training z is drawn from the posterior q(z|x), whereas
        during sampling z is drawn from the prior p(z).
        """

        if activations is None:
            return self.sample(tensor, temp)

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

    def sample(self, tensor, temp=0):
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

    #this is overridden and renamed from "input" to "in_tensor"
    def forward(self, in_tensor): # pylint: disable=arguments-renamed
        """
        Perform the forward pass through the convolution module.
        """

        out_tensor = super().forward(in_tensor)

        #optional residual gradient flow
        if self.res:
            out_tensor = out_tensor + in_tensor

        return out_tensor


class DmllNet(nn.Module):
    """
    Discrete Mixed Logistic Loss network.
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

        #the discrete logistics generate samples of shape N x n_mixtures x H x W
        #expand/repeat the target tensor along the singleton color channels, maintain 4 dimensions
        target_r = target[:, 0:1].expand(-1, self.n_mixtures, -1, -1)
        target_g = target[:, 1:2].expand(-1, self.n_mixtures, -1, -1)
        target_b = target[:, 2:3].expand(-1, self.n_mixtures, -1, -1)
        #shape N x n_mixtures x H x W

        #get the distibutions/logits from the decoder output
        dlog_r, dlog_g, dlog_b, logits = self.get_distributions(dec_out, target_r, target_g)

        #get the log probabilities of the target given the dists with current parameters
        log_prob_r = dlog_r.log_prob(target_r)
        log_prob_g = dlog_g.log_prob(target_g)
        log_prob_b = dlog_b.log_prob(target_b)
        #shape N x n_mixtures x H x W

        #sum the log probs of each color channel and scale by the softmax of the mixture logits
        log_prob = log_prob_r + log_prob_g + log_prob_b + log_softmax(logits, dim=1)
        #shape N x n_mixtures x H x W

        #exponentiate, sum, take log along the dimension of each dist
        log_prob = logsumexp(log_prob, dim=1)
        #shape N x H x W

        #scale by the number of elements per batch item
        nll = -log_prob.sum(dim=(1, 2)) / target[0].numel()
        #shape N

        #return the negative log likelihood suitable for minimization
        return nll

    def get_distributions(self, dec_out, target_r=None, target_g=None):
        """
        Get the distributions for training/sampleing.
        """

        #use a numerical stability term to prevent very small scales after exponentiation
        stability = -7

        #dist parameters for the red sub-pixel distributions
        r_mean = self.r_mean(dec_out)
        r_logscale = self.r_logscale(dec_out).clamp(min=stability)
        #shape N x n_mixtures x H x W

        #dist parameters for the blue sub-pixel distributions
        g_mean = self.g_mean(dec_out)
        g_logscale = self.g_logscale(dec_out).clamp(min=stability)
        #shape N x n_mixtures x H x W

        #green-red mixing coefficient
        gr_coeff = tanh(self.gr_coeff(dec_out))
        #shape N x n_mixtures x H x W

        #dist parameters for the green sub-pixel distributions
        b_mean = self.b_mean(dec_out)
        b_logscale = self.b_logscale(dec_out).clamp(min=stability)
        #shape N x n_mixtures x H x W

        #blue-red/blue-green mixing coefficient
        br_coeff = tanh(self.br_coeff(dec_out))
        bg_coeff = tanh(self.bg_coeff(dec_out))
        #shape N x n_mixtures x H x W

        if target_r is None or target_g is None:
            #target is None when sampling
            #g_mean/b_mean are mixed with the distribution means for conditional sampling

            #mix the mean of green sub-pixel with red
            g_mean = g_mean + (gr_coeff * r_mean)
            #shape N x n_mixtures x H x W

            #mix the mean of blue sub-pixel with red/green
            b_mean = b_mean + (br_coeff * r_mean) + (bg_coeff * g_mean)
            #shape N x n_mixtures x H x W

        else:
            #target is non-None when training
            #g_mean/b_mean are mixed with target values to model sub-pixel conditional dependencies

            #mix the mean of green sub-pixel with red
            g_mean = g_mean + (gr_coeff * target_r)
            #shape N x n_mixtures x H x W

            #mix the mean of blue sub-pixel with red/green
            b_mean = b_mean + (br_coeff * target_r) + (bg_coeff * target_g)
            #shape N x n_mixtures x H x W

        #initialize the distributions
        dlog_r = DiscreteLogistic(r_mean, torch.exp(r_logscale), self.bits)
        dlog_g = DiscreteLogistic(g_mean, torch.exp(g_logscale), self.bits)
        dlog_b = DiscreteLogistic(b_mean, torch.exp(b_logscale), self.bits)

        #log probability of each mixture for each distribution
        logits = self.logits(dec_out)
        #shape N x n_mixtures x H x W

        return dlog_r, dlog_g, dlog_b, logits

    def sample(self, dec_out):
        """
        Sample from the discrete logistic distribution.
        """

        #get the distibutions/distribution-log-probabilities from the decoder output
        dlog_r, dlog_g, dlog_b, logits = self.get_distributions(dec_out)

        #get the color channel samples from all n_mixtures
        color_r = dlog_r.sample()
        color_g = dlog_g.sample()
        color_b = dlog_b.sample()
        #shape N x n_mixtures x H x W

        #randomly choose 1 of n_mixtures distributions per-pixel, based on their log probabilities
        #torch Categorical treats the last dim as the category
        #permute the n_mixtures dimension to the final dimension and sample
        indexes = Categorical(logits=logits.permute(0, 2, 3, 1)).sample()
        #shape N x H x W, value: an int in [0, n_mixtures - 1]

        #one hot encode the final dimension and permute to mixture dimension
        indexes = one_hot(indexes).permute(0, 3, 1, 2)
        #shape N x n_mixtures x H x W

        #indexes now has a value of 1 in the channel corresponding with the selected distribution
        #all others are zero
        #pointwise multiply with the color samples and sum along the channels axis
        #to zeroize all channels except those of the selected distributions
        color_r = (color_r * indexes).sum(dim=1, keepdim=True)
        color_g = (color_g * indexes).sum(dim=1, keepdim=True)
        color_b = (color_b * indexes).sum(dim=1, keepdim=True)
        #color_* shape N x 1 x H x W

        #stack the color channels
        #clamping to the valid range is consistent with attributing the probability of -1/1 the
        #remaining probability density toward -inf/inf
        img = torch.cat((color_r, color_g, color_b), dim=1).clamp(-1, 1)
        #shape N x 3 x H x W

        return img


class DiscreteLogistic(TransformedDistribution):
    """
    Discretized Logistic distribution.  Models values in the range [-1, 1] discretized into
    2**n_bits + 1 buckets.  Probability density outside of [-1, 1] is attributed to the upper/lower
    value as appropriate.  Central values are given density:

    CDF(val + half bucket width) - CDF(val - half bucket width)

    If a value has extremely low probability, it is assigned PDF(val) for numerical stability.
    """

    def __init__(self, mean, scale, bits=8):
        #bits parameterizes the width of the discretization bucket
        #higher bits -> smaller buckets
        #half of the width of the bucket
        self.half_width = (1 / ((2**bits) - 1))

        #this continuous logistic snippet adapted from pytorch distribution docs
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
        #this value is used when prob < prob_threshold
        #this would indicate the distribution parameters are off by quite a bit
        log_prob = super().log_prob(value)

        #find the discrete non-log probability of laying within the bucket
        prob = (self.cdf(value + self.half_width) - self.cdf(value - self.half_width))

        #if the non-log probability is above a threshold,
        #replace continuous log-probability with the discrete log-probability
        mask = prob > prob_threshold
        log_prob[mask] = prob[mask].clamp(min=stability).log()

        #edge case at -1:  replace -1 with cdf(-1 + half bucket width)
        mask = value <= -1
        log_prob[mask] = self.cdf(value + self.half_width)[mask].clamp(min=stability).log()

        #edge case at 1:  replace 1 with (1 - cdf(1 - half bucket width))
        mask = value >= 1
        log_prob[mask] = (1 - self.cdf(value - self.half_width))[mask].clamp(min=stability).log()

        return log_prob

    def entropy(self):
        """
        Not required for this use case.
        """

        raise NotImplementedError

    def enumerate_support(self, expand=True):
        """
        Not required for this use case.
        """

        raise NotImplementedError

    @property
    def mean(self):
        """
        Not required for this use case.
        """

        raise NotImplementedError

    @property
    def mode(self):
        """
        Not required for this use case.
        """

        raise NotImplementedError

    @property
    def variance(self):
        """
        Not required for this use case.
        """

        raise NotImplementedError
