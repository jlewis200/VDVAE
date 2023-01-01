#!/usr/bin/python3


import unittest
import torch

from pathlib import Path
from configs import get_model
from torch.nn.functional import mse_loss


#get the path of the source file (not the CWD)
PATH = str(Path(__file__).parent) + '/'
DEVICE = torch.device("cuda:0")


class test_encoder(unittest.TestCase):
    """
    Test the encoder.
    """

    def test_encoder_activations_cifar10(self):
        """
        Test the encoder.

        The tensors were extracted from a training batch of the original VDVAE.
        The regression test ensures the activations of the refactored encoder match the original VDVAE.
        This ensures:
            -the weights from the pretrained encoder were transferred correctly
            -the encoder uses the proper configuration of layers
        """

        #the input to the VDVAE encoder network
        batch = torch.load(PATH + "tensors/cifar10_batch.pt").to(DEVICE)

        #original VDVAE uses TF standard N x H x W x C
        #this implementation uses pytorch standard N x C x H x W
        batch = batch.permute(0, 3, 1, 2) 

        #the layer activations from the original VDVAE
        expected_activations = torch.load(PATH + "tensors/cifar10_encoder_activations.pt")

        #initialize the pretrained model
        model = get_model("cifar10").to(DEVICE)
        checkpoint = torch.load("checkpoints/cifar10_pretrained.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        #pass the batch through the encoder
        model_activations = model.encoder.forward(batch)
       
        #ensure dictionaries are the same length
        self.assertEqual(len(model_activations), len(expected_activations))

        #ensure the model activations matche the VDVAE activations
        for key in expected_activations.keys():
            self.assertTrue(torch.allclose(model_activations[key], 
                                           expected_activations[key].to(DEVICE), 
                                           atol=1e-4))

