from gpupy.gpupy import Gpupy
import numpy as np

class test_conv_2d():

    def setup(self):
        self.gp = Gpupy()
        self.rng = np.random.RandomState(0)

    def test_conv_2d(self):
        """ Test 2D convolution of 4D tensors."""
        inputs = np.array(self.rng.rand(2, 3, 2, 2), dtype=np.float32, order='F')
        kernels = np.array(self.rng.rand(3, 3, 1, 1), dtype=np.float32, order='F')

        out_gp = self.gp.conv_2d(inputs, kernels).copy_to_host()
