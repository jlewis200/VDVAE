#!/usr/bin/python3

import torch
import numpy as np

from models import VAE

from torchvision.datasets import CelebA, CIFAR10
from torch.utils.data import TensorDataset
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, RandomHorizontalFlip, Normalize

def get_model(config):
    """
    """

    if config == "cifar10":
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

        model = VAE(encoder_layers, decoder_layers)
        dataset_kwargs = {"root": "cifar10", 
                          "download": True, 
                          "transform": ToTensor()}
        model.get_dataset = lambda: CIFAR10(dataset_kwargs)
       
        #mean/std used by VDVAE to scale model input
        mean = 0.473091686
        std = 0.251636706
        model.transform_in = Compose((Normalize(mean, std), Resize((32, 32))))

        #mean/std used by VDVAE to scale targets for the loss function
        #map [0, 1] to [-1, 1]
        model.transform_target = Compose((Normalize(0.5, 0.5), Resize((32, 32))))

    elif config == "ffhq256":
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

        model = VAE(encoder_layers, decoder_layers, low_bit=True)

        #use a lambda so we can set the config but delay loading until we're sure we want to train
        model.get_dataset = lambda: TensorDataset(torch.from_numpy(np.load("ffhq256/ffhq-256.npy")))
       
        #mean/std used by VDVAE to scale model input
        mean = 0.440885452
        std = 0.272842979
        model.transform_in = Compose((Normalize(mean, std), Resize((256, 256))))

        #mean/std used by VDVAE to scale targets for the loss function
        #map [0, 1] to [-1, 1]
        model.transform_target = Compose((Normalize(0.5, 0.5), Resize((256, 256))))
        
    elif config == "celeba":
        encoder_layers = [
            {"channels": 512, "n_blocks":  3, "resolution": 128},
            {"channels": 512, "n_blocks":  8, "resolution":  64},
            {"channels": 512, "n_blocks": 12, "resolution":  32},
            {"channels": 512, "n_blocks": 17, "resolution":  16},
            {"channels": 512, "n_blocks":  7, "resolution":   8},
            {"channels": 512, "n_blocks":  5, "resolution":   4},
            {"channels": 512, "n_blocks":  4, "resolution":   1}]   

        decoder_layers = [
            {"channels": 512, "n_blocks":  2, "resolution":   1},
            {"channels": 512, "n_blocks":  3, "resolution":   4},
            {"channels": 512, "n_blocks":  4, "resolution":   8},
            {"channels": 512, "n_blocks":  9, "resolution":  16},
            {"channels": 512, "n_blocks": 21, "resolution":  32},
            {"channels": 512, "n_blocks": 13, "resolution":  64, "bias": False},
            {"channels": 512, "n_blocks":  7, "resolution": 128, "bias": False}]

        model = VAE(encoder_layers, decoder_layers)
        model.dataset = CelebA
        mean = 0.
        std = 0.
        model.transform = Compose((ToTensor(), Normalize(mean, std), Resize((128, 128))))
        model.dataset_kwargs = {"root": "celeba", 
                                "download": True, 
                                "transform": model.transform}

    else:
        print("unsupported dataset")
        return

    return model


