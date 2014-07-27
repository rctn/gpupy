import gpupy.gpupy as gpupy
from numbapro import cuda
import numpy as np

class test_check():

    def setup(self):
        self.rng = np.random.RandomState(0)

    def test_check(self):
        a = self.rng.rand(129, 1025).astype(np.float32)
        b, dtype = gpupy._check_array(a)
        assert b.dtype == np.float32
        assert isinstance(b, cuda.cudadrv.devicearray.DeviceNDArray)


