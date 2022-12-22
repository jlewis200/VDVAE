#!/usr/bin/python3

"""
Variational Auto Encoder (VAE) encoder/decoder models.
"""

import torch

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
                 n_encoder_blocks):

        super().__init__()

        #ratio per VDVAE
        mid_channels = channels // 4
        
        self.encoder_blocks = nn.Sequential()

        for _ in range(n_encoder_blocks):
            self.encoder_blocks.append(Block(channels, mid_channels, channels))
        
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
        tensor = avg_pool2d(tensor, 2)
        
        return tensor


class Block(nn.Sequential):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 res=False):

        super().__init__()

        self.res = res

        self.append(GELU())
        self.append(Conv2d(in_channels, mid_channels, 1))
        self.append(GELU())
        self.append(Conv2d(mid_channels, mid_channels, 3, padding=1))
        self.append(GELU())
        self.append(Conv2d(mid_channels, mid_channels, 3, padding=1))
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
                 bias_shape,
                 encoder_meta_block=None):
        
        super().__init__()

        self.encoder_meta_block=encoder_meta_block
        self.decode_blocks = nn.Sequential()

        for _ in range(n_decoder_blocks):
            self.decode_blocks.append(DecoderBlock(channels, encoder_meta_block=encoder_meta_block))
        
        self.activations = None
        self.bias = nn.Parameter(torch.zeros(bias_shape))

    def forward(self, tensor=None):  #output of previous decoder-meta block, or None for first decoder-meta
        """
        Perform the forward pass through the convolution module.  Store the
        output out_tensor.
        """

        if tensor is None:
            #encoder information only enters the decoding path through reparameterized sample of z
            #tensor will be None for the first decoder meta-block
            #use a zero-tensor shaped like the paired encoder layer's activations
            tensor = torch.zeros_like(self.encoder_meta_block.activations)

            if torch.cuda.is_available():
                tensor = tensor.cuda()

        else:
            #upsample the output of the previous decoder meta block
            tensor = interpolate(tensor, scale_factor=2)
       
        #add the bias term for this decoder-meta block
        tensor = tensor + self.bias
           
        return self.decode_blocks(tensor)

    def get_loss_kl(self):
        """
        Get the KL divergence loss for each decoder block in this decoder meta block.
        """
       
        #begin with a zero-tensor
        loss_kl = torch.zeros(1, requires_grad=True)

        if torch.cuda.is_available():
            loss_kl = loss_kl.cuda()

        #sum the KL losses from each decder block
        for decode_block in self.decode_blocks:
            loss_kl = loss_kl + (decode_block.loss_kl / len(self.decode_blocks))

        return loss_kl


class DecoderBlock(nn.Module):
    """
    Decoder block per VDVAE architecture.
    """

    def __init__(self,
                 channels,
                 encoder_meta_block=None):
        
        super().__init__()

        #store a reference to the encoder_meta_block at this tensor resolution
        self.encoder_meta_block = encoder_meta_block

        #mid_channels and z_channels ratios from VDVAE
        mid_channels = channels // 4
        self.z_channels = channels // 32
       
        #block ratios from VDVAE
        self.phi = Block(channels * 2, mid_channels, 2 * self.z_channels)
        self.theta = Block(channels, mid_channels, channels + (2 * self.z_channels))
        self.z_projection = Conv2d(self.z_channels, channels, kernel_size=1)
        self.res = Block(channels, mid_channels, channels, res=True)


    def forward(self,
                tensor):            #output of previous decoder block
        """
        Perform the forward pass through the convolution module.  Store the
        output out_tensor.
        """

        #get the activations from the paired-encoder block
        enc_activations = self.encoder_meta_block.activations

        #get the mean/log-variance of q
        q_mean, q_logvar = self.phi(torch.cat((tensor, enc_activations), dim=1)).chunk(2, dim=1)
        
        #get the mean, log-variance of p, and the residual flow to the main path
        theta = self.theta(tensor)
        p_mean, p_logvar, res = torch.tensor_split(theta, (self.z_channels, 2 * self.z_channels), dim=1)

        #interact(local=locals()) 
        #join the output from theta with the main branch
        tensor = tensor + res

        #get q = N(q_mean, q_std)
        q_dist = torch.distributions.Normal(q_mean, torch.exp(q_logvar / 2))

        #get reparameterized sample from N(q_mean, q_std)
        q_rsample = q_dist.rsample()

        #get p = N(p_mean, p_std)
        p_dist = torch.distributions.Normal(p_mean, torch.exp(p_logvar / 2))

        #calculate the KL divergence between p and q
        self.loss_kl = (q_dist.log_prob(q_rsample) - p_dist.log_prob(q_rsample)).mean().abs()

        tensor = tensor + self.z_projection(q_rsample)

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
            {"channels":256, "n_encoder_blocks":1},
            {"channels":256, "n_encoder_blocks":1},
            {"channels":256, "n_encoder_blocks":1},
            {"channels":256, "n_encoder_blocks":1},
            {"channels":256, "n_encoder_blocks":1},
            {"channels":256, "n_encoder_blocks":1},]

        self.encoder = nn.Sequential()
        self.encoder.append(Conv2d(3, 256, kernel_size=3, padding=1))

        for encoder_layer in encoder_layers:
            self.encoder.append(EncoderMetaBlock(**encoder_layer))
        
        ######################################################################## 
        #decoder

        decoder_layers = [
            {"channels":256, "n_decoder_blocks":1, "bias_shape":(1, 256,   4,   4)},
            {"channels":256, "n_decoder_blocks":1, "bias_shape":(1, 256,   8,   8)},
            {"channels":256, "n_decoder_blocks":1, "bias_shape":(1, 256,  16,  16)},
            {"channels":256, "n_decoder_blocks":1, "bias_shape":(1, 256,  32,  32)},
            {"channels":256, "n_decoder_blocks":1, "bias_shape":(1, 256,  64,  64)},
            {"channels":256, "n_decoder_blocks":1, "bias_shape":(1, 256, 128, 128)}]

        self.decoder = nn.Sequential()
        
        for decoder_layer, encoder_meta_block in zip(decoder_layers, self.encoder[::-1]):
            self.decoder.append(DecoderMetaBlock(**decoder_layer, encoder_meta_block=encoder_meta_block))

        self.decoder.append(Conv2d(256, 3, kernel_size=3, padding=1))

    def encode(self, tensor):
        """
        Pass input tensor through the encoder.
        """
    
        return self.encoder.forward(tensor)


    def decode(self, tensor):
        """
        Pass latent encoding tensor through the decoder.
        """

        return self.decoder.forward(tensor)


    def get_loss_kl(self):
        loss_kl = torch.zeros(1, requires_grad=True)

        if torch.cuda.is_available():
            loss_kl = loss_kl.cuda()

        for decode_meta_block in self.decoder:
            if isinstance(decode_meta_block, DecoderMetaBlock):
                #TODO:  fix the scaling, right now theres an output convolution in the decoder as well 
                #TODO:  when that conv is removed, there won't be any need for the isinstance check
                #scale by number of decoder-meta blocks
                loss_kl = loss_kl + (decode_meta_block.get_loss_kl() / len(self.decoder))

        return loss_kl


