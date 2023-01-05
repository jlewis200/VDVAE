#!/usr/bin/python3


import unittest
import torch

from vdvae import main, get_args 


class test_experiments(unittest.TestCase):
    """
    Test the VDVAE experiments.
    """

    def test_reconstruct(self):
        """
        Test the image reconstruction function.
        """

        arg_list = ["--checkpoint", "checkpoints/cifar10_pretrained.pt",
                    "--config", "cifar10",
                    "--reconstruct", "tests/imgs/reconstruct.jpg",
                    "--device", "cpu"]

        args = get_args(arg_list)

        main(args)


    def test_sample(self):
        """
        Test the sample function.
        """

        arg_list = ["--checkpoint", "checkpoints/cifar10_pretrained.pt",
                    "--config", "cifar10",
                    "--sample", "6",
                    "--device", "cpu"]

        args = get_args(arg_list)

        main(args)


