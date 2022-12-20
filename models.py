#!/usr/bin/python3

"""
Variational Auto Encoder (VAE) encoder/decoder models.
"""

from torch import nn
from torch.nn.functional import interpolate, avg_pool2d

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

class ConvModule(nn.Sequential):
    """
    Convolutional Module for use in the encoder.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=2,
                 momentum=0.1,
                 affine=True,
                 negative_slope=0.01):

        super().__init__()
        
        self.append(nn.GELU())
        self.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
        #self.append(nn.BatchNorm2d(out_channels, momentum=momentum, affine=affine))

    def forward(self, in_tensor):
        """
        Perform the forward pass through the convolution module.  Store the
        output out_tensor.
        """

        out_tensor = super().forward(in_tensor)
        self.out_tensor = out_tensor

        return out_tensor


class ConvTransposeModule(nn.Sequential):
    """
    Transposed convolutional Module for use in the decoder.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=2,
                 output_padding=1,
                 momentum=0.1,
                 affine=True,
                 negative_slope=0.01):

        super().__init__()

        self.append(nn.GELU())
        self.append(nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=output_padding))

        #self.append(nn.BatchNorm2d(out_channels, momentum=momentum, affine=affine))


    def forward(self, in_tensor):
        """
        Perform the forward pass through the convolution module.  Store the
        output out_tensor.
        """

        out_tensor = super().forward(in_tensor)
        self.out_tensor = out_tensor

        return out_tensor


class VAE(nn.Module):
    """
    VAE Encoder/Decoder model.
    """

    def __init__(self):
        super().__init__()
        bias = True

        out_size = 1024
        flat_size = 4096

        self.encoder = nn.Module()
        self.decoder = nn.Sequential()

        #encoder
        self.encoder.convs = nn.Sequential()
        self.encoder.flatten = nn.Sequential()
        self.encoder.mean = nn.Sequential()
        self.encoder.var = nn.Sequential()

        in_channels = 3
        out_channels = 32

        #shape:  N x 3 x 128 x 128

        for _ in range(6):
            self.encoder.convs.append(ConvModule(in_channels, out_channels))
            in_channels = out_channels
            out_channels *= 2

        #shape:  N x 1024 x 2 x 2

        self.encoder.flatten.append(nn.Flatten())
        self.encoder.flatten.append(nn.Linear(flat_size, flat_size, bias=bias))
        self.encoder.flatten.append(nn.Dropout(p=0.05))

        #shape: N x 4096

        #in: N x 4096
        self.encoder.mean.append(nn.Linear(flat_size, out_size, bias=bias))
        #out: N x 1024

        #in: N x 4096
        self.encoder.var.append(nn.Linear(flat_size, out_size, bias=bias))
        #out: N x 1024

        #decoder
        self.decoder.unflatten = nn.Sequential()
        self.decoder.convs = nn.Sequential()
        
        #self.out = nn.Sequential()

        #shape:  N x 1024

        self.decoder.unflatten.append(nn.Linear(1024, flat_size, bias=bias))
        self.decoder.unflatten.append(nn.Dropout(p=0.05))

        #shape:  N x 4096

        self.decoder.unflatten.append(View((1024, 2, 2)))

        #shape:  N x 1024 x 2 x 2

        in_channels = 1024
        out_channels = 512

        for _ in range(5):
            self.decoder.convs.append(ConvTransposeModule(in_channels, out_channels))

            in_channels = out_channels
            out_channels //= 2
            out_channels = max(32, out_channels)

        #shape:  N x 32 x 128 x 128

        self.decoder.convs.append(ConvTransposeModule(32, 3))

        #shape:  N x 3 x 128 x 128

    def encode(self, tensor):
        """
        Pass input tensor through the convs portion of the encoder.  Pass the intermediate output
        through the mean/var branches.
        """

        #get intermediate flattened encoding
        intermediate = self.encoder.convs.forward(tensor)
        intermediate = self.encoder.flatten.forward(intermediate)

        #pass to mean/log_var branches
        mean = self.encoder.mean.forward(intermediate)
        var = self.encoder.var.forward(intermediate)

        #out (N x 1024, N x 1024)
        return mean, var


    def decode(self, tensor):
        """
        Pass latent encoding tensor through the decoder.
        """

        return self.decoder.forward(tensor)
        

class Block(nn.Sequential):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 up_sample=False,
                 down_sample=False):

        super().__init__()

        self.up_sample = up_sample
        self.down_sample = down_sample
        
        self.append(nn.GELU())
        self.append(nn.Conv2d(in_channels, mid_channels, 1))
        self.append(nn.GELU())
        self.append(nn.Conv2d(mid_channels, mid_channels, 3, padding=1))
        self.append(nn.GELU())
        self.append(nn.Conv2d(mid_channels, mid_channels, 3, padding=1))
        self.append(nn.GELU())
        self.append(nn.Conv2d(mid_channels, out_channels, 1))


    def forward(self, tensor):
        """
        Perform the forward pass through the convolution module.  Store the
        output out_tensor.
        """

        if self.up_sample:
            tensor = interpolate(tensor, scale_factor=2)
        
        tensor = super().forward(tensor)
        
        if self.down_sample:
            tensor = avg_pool2d(tensor, 2)

        self.out_tensor = tensor

        return tensor


class VAEAvgPool(nn.Module):
    """
    VAE Encoder/Decoder using stride 1 convolutions and avgpool/interpolate
    within the encoder/decoder.
    """

    def __init__(self):
        super().__init__()
        bias = True

        flat_size = 1024
        out_size = 1024

        self.encoder = nn.Module()
        self.decoder = nn.Sequential()

        #encoder
        self.encoder.convs = nn.Sequential()
        self.encoder.flatten = nn.Sequential()
        self.encoder.mean = nn.Sequential()
        self.encoder.var = nn.Sequential()

        #shape:  N x 3 x 128 x 128

        channels = [(  3, 256, 256),
                    (256, 256, 256),
                    (256, 256, 256),
                    (256, 256, 256),
                    (256, 256, 256),
                    (256, 256, 256)]

        for channel in channels:
            self.encoder.convs.append(Block(*channel, down_sample=True))

        #shape:  N x 256 x 2 x 2

        self.encoder.flatten.append(nn.Flatten())

        #shape: N x 1024

        #in: N x 1024
        self.encoder.mean.append(nn.Linear(flat_size, out_size, bias=bias))
        #out: N x 1024

        #in: N x 1024
        self.encoder.var.append(nn.Linear(flat_size, out_size, bias=bias))
        #out: N x 1024

        #decoder
        self.decoder.unflatten = nn.Sequential()
        self.decoder.convs = nn.Sequential()
        
        #self.out = nn.Sequential()

        #shape:  N x 1024

        self.decoder.unflatten.append(nn.Linear(out_size, flat_size, bias=bias))

        #shape:  N x 1024

        self.decoder.unflatten.append(View((256, 2, 2)))

        #shape:  N x 256 x 2 x 2

        #traverse channel tuples in reverse order, also reverse channel entries
        for channel in channels[::-1]:
            self.decoder.convs.append(Block(*channel[::-1], up_sample=True))

        #shape:  N x 3 x 128 x 128


    def encode(self, tensor):
        """
        Pass input tensor through the convs portion of the encoder.  Pass the intermediate output
        through the mean/var branches.
        """

        #get intermediate flattened encoding
        intermediate = self.encoder.convs.forward(tensor)
        intermediate = self.encoder.flatten.forward(intermediate)

        #pass to mean/log_var branches
        mean = self.encoder.mean.forward(intermediate)
        var = self.encoder.var.forward(intermediate)

        #out (N x 1024, N x 1024)
        return mean, var


    def decode(self, tensor):
        """
        Pass latent encoding tensor through the decoder.
        """

        return self.decoder.forward(tensor)
