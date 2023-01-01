#!/usr/bin/python3


import unittest
import torch

from pathlib import Path
from configs import get_model
from torch.nn.functional import mse_loss
from models import Block

#get the path of the source file (not the CWD)
PATH = str(Path(__file__).parent) + '/'
DEVICE = torch.device("cuda:0")


class test_block(unittest.TestCase):
    """
    Test the basic res block.
    """

    def test_block_kernel_3_res(self):
        """
        Test a block configured with kernel size 1 and a residual path.
        """

        block = Block(in_channels=2,
                      mid_channels=1,
                      out_channels=2,
                      res=True,
                      mid_kernel=3,
                      final_scale=1)

        #set the convolution weights/biases to known values
        for idx in range(1, 8, 2):
            block[idx].weight.data.fill_(1.0)
            block[idx].bias.data.fill_(0.0)
        
        #fill with 5 to keep GELU out of non-linear region
        tensor = torch.full((1, 2, 8, 9), 5.0)

        #expected value calculation:
        #tensor shape 1 x 2 x 32 x 32
        
        #conv in_chan 2, out_chan 1, kernel 1
        #with 2 filters and 1 x 1 kernels the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + (kernel_1 x tensor) + bias
        #(1 * 5) + (1 * 5) = 10
        
        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 1, kernel 3
        #assuming a tensor filled with same values
        #3 x 3 kernel (values 1), zero bias, padding size 1, padding mode zeros the pointwise tensor value is
        #corner cases:  4 * tensor
        #edge cases:  6 * tensor
        #central cases:  9 * tensor

        #  40 60 60 ... 60 60 40
        #  60 90 90 ... 90 90 60
        #  60 90 90 ... 90 90 60
        #   ...             ...
        #  60 90 90 ... 90 90 60
        #  60 90 90 ... 90 90 60
        #  40 60 60 ... 60 60 40


        #conv in_chan 1, out_chan 1, kernel 3
        #assuming a tensor filled with same values
        #3 x 3 kernel (values 1), zero bias, padding size 1, padding mode zeros the pointwise tensor value is
        #corner cases:  4 * tensor
        #edge cases:  6 * tensor
        #central cases:  9 * tensor

        # 250 400 450 450 ... 450 450 400 250
        # 400 640 720 720 ... 720 720 640 400
        # 450 720 810 810 ... 810 810 720 450
        # 450 720 810 810 ... 810 810 720 450
        # ...                             ...
        # 450 720 810 810 ... 810 810 720 450
        # 450 720 810 810 ... 810 810 720 450
        # 400 640 720 720 ... 720 720 640 400
        # 250 400 450 450 ... 450 450 400 250
        
        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 1, kernel 1
        #with 1 filters and 1 x 1 kernel the pointwise tensor value is
        #tensor value (kernel_0 x tensor)
        
        #tensor shape 1 x 2 x 32 x 32

        # 250 400 450 450 ... 450 450 400 250
        # 400 640 720 720 ... 720 720 640 400
        # 450 720 810 810 ... 810 810 720 450
        # 450 720 810 810 ... 810 810 720 450
        # ...                             ...
        # 450 720 810 810 ... 810 810 720 450
        # 450 720 810 810 ... 810 810 720 450
        # 400 640 720 720 ... 720 720 640 400
        # 250 400 450 450 ... 450 450 400 250
        

        tensor = block.forward(tensor)
        expected_tensor = torch.tensor(
            [[250, 400, 450, 450, 450, 450, 450, 400, 250],
             [400, 640, 720, 720, 720, 720, 720, 640, 400],
             [450, 720, 810, 810, 810, 810, 810, 720, 450],
             [450, 720, 810, 810, 810, 810, 810, 720, 450],
             [450, 720, 810, 810, 810, 810, 810, 720, 450],
             [450, 720, 810, 810, 810, 810, 810, 720, 450],
             [400, 640, 720, 720, 720, 720, 720, 640, 400],
             [250, 400, 450, 450, 450, 450, 450, 400, 250]]).to(tensor.dtype)
        expected_tensor = expected_tensor.unsqueeze(0)
        expected_tensor = expected_tensor.unsqueeze(0)
        #shape 1 x 1 x 8 x 9
        
        #duplicate across channel dimension
        expected_tensor = expected_tensor.expand(-1, 2, -1, -1)
        #shape 1 x 2 x 8 x 9

        #add the residual from input tensor
        expected_tensor = expected_tensor + 5

        #ensure shape is expected shape
        self.assertEqual(tensor.shape, expected_tensor.shape)

        #ensure value is expected value
        self.assertTrue(torch.allclose(tensor, expected_tensor))

    def test_block_kernel_3_no_res(self):
        """
        Test a block configured with kernel size 1 and no residual path.
        """

        block = Block(in_channels=2,
                      mid_channels=1,
                      out_channels=2,
                      res=False,
                      mid_kernel=3,
                      final_scale=1)

        #set the convolution weights/biases to known values
        for idx in range(1, 8, 2):
            block[idx].weight.data.fill_(1.0)
            block[idx].bias.data.fill_(0.0)
        
        #fill with 5 to keep GELU out of non-linear region
        tensor = torch.full((1, 2, 8, 9), 5.0)

        #expected value calculation:
        #tensor shape 1 x 2 x 32 x 32
        
        #conv in_chan 2, out_chan 1, kernel 1
        #with 2 filters and 1 x 1 kernels the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + (kernel_1 x tensor) + bias
        #(1 * 5) + (1 * 5) = 10
        
        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 1, kernel 3
        #assuming a tensor filled with same values
        #3 x 3 kernel (values 1), zero bias, padding size 1, padding mode zeros the pointwise tensor value is
        #corner cases:  4 * tensor
        #edge cases:  6 * tensor
        #central cases:  9 * tensor

        #  40 60 60 ... 60 60 40
        #  60 90 90 ... 90 90 60
        #  60 90 90 ... 90 90 60
        #   ...             ...
        #  60 90 90 ... 90 90 60
        #  60 90 90 ... 90 90 60
        #  40 60 60 ... 60 60 40


        #conv in_chan 1, out_chan 1, kernel 3
        #assuming a tensor filled with same values
        #3 x 3 kernel (values 1), zero bias, padding size 1, padding mode zeros the pointwise tensor value is
        #corner cases:  4 * tensor
        #edge cases:  6 * tensor
        #central cases:  9 * tensor

        # 250 400 450 450 ... 450 450 400 250
        # 400 640 720 720 ... 720 720 640 400
        # 450 720 810 810 ... 810 810 720 450
        # 450 720 810 810 ... 810 810 720 450
        # ...                             ...
        # 450 720 810 810 ... 810 810 720 450
        # 450 720 810 810 ... 810 810 720 450
        # 400 640 720 720 ... 720 720 640 400
        # 250 400 450 450 ... 450 450 400 250
        
        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 1, kernel 1
        #with 1 filters and 1 x 1 kernel the pointwise tensor value is
        #tensor value (kernel_0 x tensor)
        
        #tensor shape 1 x 2 x 32 x 32

        # 250 400 450 450 ... 450 450 400 250
        # 400 640 720 720 ... 720 720 640 400
        # 450 720 810 810 ... 810 810 720 450
        # 450 720 810 810 ... 810 810 720 450
        # ...                             ...
        # 450 720 810 810 ... 810 810 720 450
        # 450 720 810 810 ... 810 810 720 450
        # 400 640 720 720 ... 720 720 640 400
        # 250 400 450 450 ... 450 450 400 250
        

        tensor = block.forward(tensor)
        expected_tensor = torch.tensor(
            [[250, 400, 450, 450, 450, 450, 450, 400, 250],
             [400, 640, 720, 720, 720, 720, 720, 640, 400],
             [450, 720, 810, 810, 810, 810, 810, 720, 450],
             [450, 720, 810, 810, 810, 810, 810, 720, 450],
             [450, 720, 810, 810, 810, 810, 810, 720, 450],
             [450, 720, 810, 810, 810, 810, 810, 720, 450],
             [400, 640, 720, 720, 720, 720, 720, 640, 400],
             [250, 400, 450, 450, 450, 450, 450, 400, 250]]).to(tensor.dtype)
        expected_tensor = expected_tensor.unsqueeze(0)
        expected_tensor = expected_tensor.unsqueeze(0)
        #shape 1 x 1 x 8 x 9
        
        #duplicate across channel dimension
        expected_tensor = expected_tensor.expand(-1, 2, -1, -1)
        #shape 1 x 2 x 8 x 9

        #ensure shape is expected shape
        self.assertEqual(tensor.shape, expected_tensor.shape)

        #ensure value is expected value
        self.assertTrue(torch.allclose(tensor, expected_tensor))

    def test_block_kernel_1_res(self):
        """
        Test a block configured with kernel size 1 and a residual path.
        """

        block = Block(in_channels=2,
                      mid_channels=1,
                      out_channels=2,
                      res=True,
                      mid_kernel=1,
                      final_scale=1)

        #set the convolution weights/biases to known values
        for idx in range(1, 8, 2):
            block[idx].weight.data.fill_(2.0)
            block[idx].bias.data.fill_(1.0)
        
        #fill with 10 to keep GELU out of non-linear region
        tensor = torch.full((1, 2, 32, 32), 10.0)
        
        #expected value calculation:
        #tensor shape 1 x 2 x 32 x 32
        
        #conv in_chan 2, out_chan 1, kernel 1
        #with 2 filters and 1 x 1 kernels the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + (kernel_1 x tensor) + bias
        #(2 * 10) + (2 * 10) + 1 = 41
        
        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 1, kernel 1
        #with 1 filters and 1 x 1 kernel the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + bias
        #(41 * 2) + 1 = 83

        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 1, kernel 1
        #with 1 filters and 1 x 1 kernel the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + bias
        #(2 * 83) + 1 = 167

        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 2, kernel 1
        #with 1 filters and 1 x 1 kernel the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + bias
        #(2 x 167) + 1 = 335

        #add the residual from input tensor
        #10 + 335 = 10

        tensor = block.forward(tensor)
        expected_tensor = torch.full((1, 2, 32, 32), 345.0)

        #ensure shape is expected shape
        self.assertEqual(tensor.shape, expected_tensor.shape)

        #ensure value is expected value
        self.assertTrue(torch.allclose(tensor, expected_tensor))

    def test_block_kernel_1_no_res(self):
        """
        Test a block configured with kernel size 1 and no residual path.
        """

        block = Block(in_channels=2,
                      mid_channels=1,
                      out_channels=2,
                      res=False,
                      mid_kernel=1,
                      final_scale=1)

        #set the convolution weights/biases to known values
        for idx in range(1, 8, 2):
            block[idx].weight.data.fill_(2.0)
            block[idx].bias.data.fill_(1.0)
        
        #fill with 10 to keep GELU out of non-linear region
        tensor = torch.full((1, 2, 32, 32), 10.0)
        
        #expected value calculation:
        #tensor shape 1 x 2 x 32 x 32
        
        #conv in_chan 2, out_chan 1, kernel 1
        #with 2 filters and 1 x 1 kernels the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + (kernel_1 x tensor) + bias
        #(2 * 10) + (2 * 10) + 1 = 41
        
        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 1, kernel 1
        #with 1 filters and 1 x 1 kernel the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + bias
        #(41 * 2) + 1 = 83

        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 1, kernel 1
        #with 1 filters and 1 x 1 kernel the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + bias
        #(2 * 83) + 1 = 167

        #tensor shape 1 x 1 x 32 x 32

        #conv in_chan 1, out_chan 2, kernel 1
        #with 1 filters and 1 x 1 kernel the pointwise tensor value is
        #tensor value (kernel_0 x tensor) + bias
        #(2 x 167) + 1 = 335

        tensor = block.forward(tensor)
        expected_tensor = torch.full((1, 2, 32, 32), 335.0)

        #ensure shape is expected shape
        self.assertEqual(tensor.shape, expected_tensor.shape)

        #ensure value is expected value
        self.assertTrue(torch.allclose(tensor, expected_tensor))


