import gpupy.gpupy as gpupy
from numbapro import cuda
import numpy as np

class test_reshape():

    def setup(self):
        self.rng = np.random.RandomState(0)

    def test_reshape(self):
        a = cuda.to_device(np.array(self.rng.rand(12, 10), dtype=np.float32,
                                    order='F'))
        b = cuda.to_device(np.array(self.rng.rand(120), dtype=np.float32,
                                    order='F'))
        shape = b.shape
        strides = b.strides
        dtype = b.dtype
        c = gpupy._cu_reshape(a, shape, strides, dtype)
        assert c.dtype == np.float32
        assert c.shape == shape
        assert c.strides == strides
        assert isinstance(b, cuda.cudadrv.devicearray.DeviceNDArray)


