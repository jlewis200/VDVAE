#!/usr/bin/python3


import unittest
import torch

from pathlib import Path
from configs import get_model

#get the path of the source file (not the CWD)
PATH = str(Path(__file__).parent) + '/'
DEVICE = torch.device("cuda:0")


class test_dmll(unittest.TestCase):
    """
    Test the Discrete Mixed Logistic Loss (DMLL) network.
    """

    def test_dmll_loss(self):
        """
        Test the DMLL loss function.

        The tensors were extracted from a training batch of the original VDVAE.
        The regression test ensures the overall loss of the refactored DMLL matches the original.
        This ensures:
            -the weights from the original pretrained DMLL net were transferred correctly
            -the DiscretizedLogistic distribution produces the correct log-probabilities
        """

        #the output of the VDVAE decoder network
        dec_out = torch.load(PATH + "tensors/dmll_dec_out.pt").to(DEVICE)

        #the original input rescaled to [-1, 1]
        target = torch.load(PATH + "tensors/dmll_target.pt").to(DEVICE)

        #original VDVAE uses the TF standard N x H x W x C, we use pytorch standard N x C x H x W
        target = target.permute(0, 3, 1, 2) 

        #the negative log likelihood from the original VDVAE
        nll = torch.load(PATH + "tensors/dmll_nll.pt").to(DEVICE)

        #initialize the pretrained model
        model = get_model("cifar10").to(DEVICE)
        checkpoint = torch.load("checkpoints/cifar10_pretrained.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        #pass the decoder output and target through the model's DMLL network
        model_nll = model.decoder.get_nll(dec_out, target)

        #ensure the model nll matches the VDVAE nll
        self.assertTrue(torch.allclose(model_nll, nll, rtol=1e-4))

 
