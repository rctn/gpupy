from gpupy.gpupy import Gpupy
import numpy as np

class test_zero_diag():

    def setup(self):
        self.gp = Gpupy()
        self.rng = np.random.RandomState(0)

    def test_zero_diag(self):
        a = self.rng.rand(129, 129).astype(np.float32)

        out_np = a-np.diag(np.diag(a))
        out_gp = self.gp.zero_diag(a).copy_to_host()
        assert(np.allclose(out_np, out_gp))
