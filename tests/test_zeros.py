from gpupy import Gpupy
import numpy as np
from numbapro import cuda

class test_zeros():

    def setup(self):
        self.gp = Gpupy()

    def test_zeros(self):
        """Create array of zeros."""
        out_np = np.zeros(shape=(129,1025), dtype=np.float32)
        out_gp = self.gp.zeros(shape=(129,1025)).copy_to_host()
        assert(np.allclose(out_np, out_gp))

        out_np = np.zeros(shape=(129,1025), dtype=np.float32)
        out_gp = cuda.to_device(np.ones(shape=(129,1025), dtype=np.float32))
        self.gp.zeros(shape=(129,1025), out=out_gp)
        assert(np.allclose(out_np, out_gp.copy_to_host()))

        out_np = np.zeros(shape=129, dtype=np.float32)
        out_gp = self.gp.zeros(shape=129).copy_to_host()
        assert(np.allclose(out_np, out_gp))

        out_np = np.zeros(129, dtype=np.float32)
        out_gp = cuda.to_device(np.ones(129, dtype=np.float32))
        self.gp.zeros(shape=129, out=out_gp)
        assert(np.allclose(out_np, out_gp.copy_to_host()))
