#!/usr/bin/python3

"""
This module is used to initialize the model for the various datasets.
"""

# pylint: disable=no-member
import torch
import numpy as np

from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torchvision.transforms.functional import resize, normalize

from model import VDVAE


def get_model(config):
    """
    Initialize the model according to the supplied config string.
    """

    model = None

    if config == "cifar10":
        model = get_cifar10()

    elif config == "ffhq256":
        model = get_ffhq256()

    return model


def get_cifar10():
    """
    Get model configured for the cifar10 dataset.
    """

    encoder_layers = [
        {"channels": 384, "n_blocks": 12, "resolution":  32},
        {"channels": 384, "n_blocks":  7, "resolution":  16},
        {"channels": 384, "n_blocks":  7, "resolution":   8},
        {"channels": 384, "n_blocks":  4, "resolution":   4},
        {"channels": 384, "n_blocks":  3, "resolution":   1}]

    decoder_layers = [
        {"channels": 384, "n_blocks":  1, "resolution":   1},
        {"channels": 384, "n_blocks":  3, "resolution":   4},
        {"channels": 384, "n_blocks":  6, "resolution":   8},
        {"channels": 384, "n_blocks": 11, "resolution":  16},
        {"channels": 384, "n_blocks": 22, "resolution":  32}]

    model = VDVAE(encoder_layers, decoder_layers)
    dataset_kwargs = {"root": "cifar10",
                      "download": True,
                      "transform": ToTensor()}

    #use a lambda to set the config but delay loading until training
    model.get_dataset = lambda: CIFAR10(**dataset_kwargs)

    #mean/std used by VDVAE to scale model input
    mean = 0.473091686
    std = 0.251636706
    model.transform_in = Compose((Normalize(mean, std), Resize((32, 32))))

    #mean/std used by VDVAE to scale targets for the loss function
    #map [0, 1] to [-1, 1]
    model.transform_target = Compose((Normalize(0.5, 0.5), Resize((32, 32))))

    #map reconstruction range [-1, 1] to PIL range [0, 1]
    model.transform_out = Compose((Normalize(-1, 2), Resize((256, 256))))

    return model


def get_ffhq256():
    """
    Get model configured for the ffhq256 dataset.
    """

    encoder_layers = [
        {"channels": 512, "n_blocks":  4, "resolution": 256},
        {"channels": 512, "n_blocks":  9, "resolution": 128},
        {"channels": 512, "n_blocks": 13, "resolution":  64},
        {"channels": 512, "n_blocks": 18, "resolution":  32},
        {"channels": 512, "n_blocks":  8, "resolution":  16},
        {"channels": 512, "n_blocks":  6, "resolution":   8},
        {"channels": 512, "n_blocks":  6, "resolution":   4},
        {"channels": 512, "n_blocks":  4, "resolution":   1}]

    decoder_layers = [
        {"channels": 512, "n_blocks":  2, "resolution":   1},
        {"channels": 512, "n_blocks":  4, "resolution":   4},
        {"channels": 512, "n_blocks":  5, "resolution":   8},
        {"channels": 512, "n_blocks": 10, "resolution":  16},
        {"channels": 512, "n_blocks": 22, "resolution":  32},
        {"channels": 512, "n_blocks": 14, "resolution":  64},
        {"channels": 512, "n_blocks":  8, "resolution": 128, "bias": False},
        {"channels": 512, "n_blocks":  1, "resolution": 256, "bias": False}]

    model = VDVAE(encoder_layers, decoder_layers, bits=5)

    #use a lambda to set the config but delay loading until training
    model.get_dataset = lambda: TensorDataset(torch.from_numpy(np.load("ffhq256/ffhq-256.npy")))

    def transform_in(tensor):
        """
        Transformation to apply to the input.
        """

        #the ffhq dataset is stored as uint8 N x H x W x C
        #convert to float [0, 1] N x C x H x W
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255
            tensor = tensor.permute(0, 3, 1, 2)

        #mean/std used by VDVAE to scale model input
        mean = 0.440885452
        std = 0.272842979

        return resize(normalize(tensor, mean, std), (256, 256))

    model.transform_in = transform_in

    def transform_target(tensor):
        """
        Transformation to apply to the target.
        """

        #the ffhq dataset is stored as uint8 N x H x W x C
        #convert to float [0, 1] N x C x H x W
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255
            tensor = tensor.permute(0, 3, 1, 2)

        return resize(normalize(tensor, 0.5, 0.5), (256, 256))

    #map PIL range [0, 1] -> loss target range [-1, 1]
    model.transform_target = transform_target

    #map reconstruction range [-1, 1] to PIL range [0, 1]
    model.transform_out = Compose((Normalize(-1, 2), Resize((256, 256))))

    return model
