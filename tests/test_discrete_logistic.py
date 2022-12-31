#!/usr/bin/python3


import unittest
import math
import torch

from pathlib import Path
from configs import get_model
from models import DiscreteLogistic

#get the path of the source file (not the CWD)
PATH = str(Path(__file__).parent) + '/'
DEVICE = torch.device("cuda:0")


class test_discrete_logistic(unittest.TestCase):
    """
    Test the Discrete Logistic distribution.
    """

    def test_sample(self):
        """
        Test the discrete logistic samples have the shape and expected probability distribution.
        """
        
        torch.manual_seed(0)

        locations = (-3., 0., 3.)
        scales = (1.5, 1., 0.5)
        shapes = (torch.Size((1000, 2000)), torch.Size((2000, 1000)), torch.Size((2000, 2000)))
        
        for location, scale, shape in zip(locations, scales, shapes):
            dist = DiscreteLogistic(torch.full(shape, location), torch.full(shape, scale))
            sample = dist.sample()

            #check the shape of the sample
            self.assertEqual(sample.shape, shape)

            #check the specified location is close to the sample mean
            self.assertTrue(torch.allclose(torch.tensor(location), sample.mean(), atol=1e-3))

            #check the closed form variance specified by scale is close to the sample variance
            expected_var = (scale**2 * math.pi**2) / 3
            self.assertTrue(torch.allclose(torch.tensor(expected_var), sample.var(), atol=1e-2))

    def test_log_prob(self):
        """
        Test the discrete logistic distribution returns the correct log-probability given an observed value.
        """

        torch.manual_seed(0)

        location = 0.
        scale = 1.
        shape = torch.Size((1,))
        
        dist = DiscreteLogistic(torch.full(shape, location), torch.full(shape, scale))
        
        #closed form cumulative density function for logistic
        cdf = lambda value: 1 / (1 + math.e**-((value - location) / scale))

        #check a central value returns the discretized log_prob
        values = ( -0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9)
        
        for value in values:
            expected_log_prob = math.log(cdf(value + (1 / 255)) - cdf(value - (1 / 255)))
            log_prob = dist.log_prob(torch.tensor(value))

            self.assertTrue(torch.allclose(log_prob, torch.tensor(expected_log_prob)))

        #check a value <= -1 is given the remaining density for values in (-inf, -1]
        values = (-2., -1.5, -1.0)
        
        for value in values:
            expected_log_prob = math.log(cdf(value + (1 / 255)))
            log_prob = dist.log_prob(torch.tensor(value))

            self.assertTrue(torch.allclose(log_prob, torch.tensor(expected_log_prob)))

        #check a value >= 1 is given the remaining density for values in [1, inf)
        values = (1.0, 1.5, 2.0)
        
        for value in values:
            expected_log_prob = math.log(1 - cdf(value - (1 / 255)))
            log_prob = dist.log_prob(torch.tensor(value))

            self.assertTrue(torch.allclose(log_prob, torch.tensor(expected_log_prob)))

        
        #check a central value returns the continuous log_prob in the case of extreme distributions
        location = -100.
        scale = 1

        dist = DiscreteLogistic(torch.full(shape, location), torch.full(shape, scale))
        
        #closed form point density function for logistic
        pdf = lambda value: (math.e**-((value - location) / scale)) / \
                            (scale * (1 + math.e**-((value - location) / scale)))**2
        
        values = ( -0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9)
        
        for value in values:
            expected_log_prob = math.log(pdf(value))
            log_prob = dist.log_prob(torch.tensor(value))

            self.assertTrue(torch.allclose(log_prob, torch.tensor(expected_log_prob)))

         
