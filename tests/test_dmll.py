#!/usr/bin/python3


import unittest
import torch

from pathlib import Path
from configs import get_model
from torch.nn.functional import mse_loss


#get the path of the source file (not the CWD)
PATH = str(Path(__file__).parent) + '/'
DEVICE = torch.device("cuda:0")


class test_dmll(unittest.TestCase):
    """
    Test the Discrete Mixed Logistic Loss (DMLL) network.
    """

    def test_dmll_loss_cifar10(self):
        """
        Test the DMLL loss function.

        The tensors were extracted from a training batch of the original VDVAE.
        The regression test ensures the overall loss of the refactored DMLL matches the original.
        This ensures:
            -the weights from the original pretrained DMLL net were transferred correctly
            -the DiscretizedLogistic distribution produces the correct log-probabilities
        """

        #the output of the VDVAE decoder network
        dec_out = torch.load(PATH + "tensors/cifar10_decoder_out.pt").to(DEVICE)

        #the original input rescaled to [-1, 1]
        target = torch.load(PATH + "tensors/cifar10_target.pt").to(DEVICE)

        #original VDVAE uses TF standard N x H x W x C
        #this implementation uses pytorch standard N x C x H x W
        target = target.permute(0, 3, 1, 2)

        #the negative log likelihood from the original VDVAE
        nll = torch.load(PATH + "tensors/cifar10_dmll_nll.pt").to(DEVICE)

        #initialize the pretrained model
        model = get_model("cifar10").to(DEVICE)
        checkpoint = torch.load("checkpoints/cifar10_pretrained.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        #pass the decoder output and target through the model's DMLL network
        model_nll = model.decoder.get_nll(dec_out, target)

        #ensure the model nll matches the VDVAE nll
        self.assertTrue(torch.allclose(model_nll, nll, rtol=1e-4))

    def test_dmll_loss_ffhq256(self):
        """
        Test the DMLL loss function.

        The tensors were extracted from a training batch of the original VDVAE.
        The regression test ensures the overall loss of the refactored DMLL matches the original.
        This ensures:
            -the weights from the original pretrained DMLL net were transferred correctly
            -the DiscretizedLogistic distribution produces the correct log-probabilities
        """
        torch.manual_seed(0)

        try:
            #the output of the VDVAE decoder network
            dec_out = torch.load(PATH + "tensors/ffhq256_decoder_out.pt").to(DEVICE)

            #the original input rescaled to [-1, 1]
            target = torch.load(PATH + "tensors/ffhq256_target.pt").to(DEVICE)

            #original VDVAE uses TF standard N x H x W x C
            #this implementation uses pytorch standard N x C x H x W
            target = target.permute(0, 3, 1, 2)

        except FileNotFoundError:
            #dec_out is ~150MB, so likely too large for github/gitlab
            return

        #the negative log likelihood from the original VDVAE
        nll = torch.load(PATH + "tensors/ffhq256_dmll_nll.pt").to(DEVICE)

        #initialize the pretrained model
        model = get_model("ffhq256").to(DEVICE)
        checkpoint = torch.load("checkpoints/ffhq256_pretrained.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        #pass the decoder output and target through the model's DMLL network
        model_nll = model.decoder.get_nll(dec_out, target)

        #ensure the model nll matches the VDVAE nll
        #a more freedom is given here as this loss function has some loss of numerical precision
        #while mathematically equivalent, the two different implementations lose precision in
        #slightly different ways and these differences can accumulate
        self.assertTrue(torch.allclose(model_nll, nll, atol=3e-3))

    def test_dmll_sample_cifar10(self):
        """
        Test the DMLL sample function.

        The dec_out tensor was extracted from a training batch of the original VDVAE.
        Within the DMLL sampling function, the original VDVAE combines the logits probability
        of each of the 10 distributions with a uniform sample.  The end result is a categorical
        distribution which chooses 1 of the 10 distributions (per-pixel) according to the probability
        of each distribution captured by the logits.  This implementation uses the pytorch built-in
        categorical distribution, so the exact distribution chosen for each pixel will not match
        between implementations, although there is no perceptible difference between the images.
        For these reasons the MSE is used as the passing metric.

        The regression test ensures the overall loss of the refactored DMLL matches the original.
        This ensures:
            -the weights from the original pretrained DMLL net were transferred correctly
            -the DiscretizedLogistic distribution produces the correct samples
        """
        torch.manual_seed(0)

        try:
            #the output of the VDVAE decoder network
            dec_out = torch.load(PATH + "tensors/cifar10_decoder_out.pt").to(DEVICE)

        except FileNotFoundError:
            #dec_out is ~150MB, so likely too large for github/gitlab
            return

        #the returned sample from the original VDVAE
        sample = torch.load(PATH + "tensors/cifar10_dmll_sample.pt").to(DEVICE)

        #original VDVAE uses the TF standard N x H x W x C, we use pytorch standard N x C x H x W
        sample = sample.permute(0, 3, 1, 2)

        #initialize the pretrained model
        model = get_model("cifar10").to(DEVICE)
        checkpoint = torch.load("checkpoints/cifar10_pretrained.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        #pass the decoder output through the model's DMLL network
        model_sample = model.decoder.dmll_net.sample(dec_out)
        mse_loss(model_sample, sample)
        #ensure the model sample is sufficiently similar to the VDVAE sample
        self.assertTrue(mse_loss(model_sample, sample) < 5e-4)


    def test_dmll_sample_ffhq256(self):
        """
        Test the DMLL sample function.

        The dec_out tensor was extracted from a training batch of the original VDVAE.
        Within the DMLL sampling function, the original VDVAE combines the logits probability
        of each of the 10 distributions with a uniform sample.  The end result is a categorical
        distribution which chooses 1 of the 10 distributions (per-pixel) according to the probability
        of each distribution captured by the logits.  This implementation uses the pytorch built-in
        categorical distribution, so the exact distribution chosen for each pixel will not match
        between implementations, although there is no perceptible difference between the images.
        For these reasons the MSE is used as the passing metric.

        The regression test ensures the overall loss of the refactored DMLL matches the original.
        This ensures:
            -the weights from the original pretrained DMLL net were transferred correctly
            -the DiscretizedLogistic distribution produces the correct samples
        """
        torch.manual_seed(0)

        try:
            #the output of the VDVAE decoder network
            dec_out = torch.load(PATH + "tensors/ffhq256_decoder_out.pt").to(DEVICE)

        except FileNotFoundError:
            #dec_out is ~150MB, so likely too large for github/gitlab
            return

        #the returned sample from the original VDVAE
        sample = torch.load(PATH + "tensors/ffhq256_dmll_sample.pt").to(DEVICE)

        #original VDVAE uses the TF standard N x H x W x C, we use pytorch standard N x C x H x W
        sample = sample.permute(0, 3, 1, 2)

        #initialize the pretrained model
        model = get_model("ffhq256").to(DEVICE)
        checkpoint = torch.load("checkpoints/ffhq256_pretrained.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        #pass the decoder output through the model's DMLL network
        model_sample = model.decoder.dmll_net.sample(dec_out)

        #ensure the model sample is sufficiently similar to the VDVAE sample
        self.assertTrue(mse_loss(model_sample, sample) < 5e-4)


