from gpupy import Gpupy
import numpy as np

class test_scal():

    def setup(self):
        self.gp = Gpupy()
        self.rng = np.random.RandomState(0)

    def test_mscal(self):
        a = self.rng.rand(129, 1025).astype(np.float32)
        alpha = 2.

        out_np = (alpha*a).astype(np.float32)
        out_gp = self.gp.scal(a, alpha).copy_to_host()

        assert(np.allclose(out_np, out_gp))

    def test_vscal(self):
        a = self.rng.rand(1025).astype(np.float32)
        alpha = 2.

        out_np = (alpha*a).astype(np.float32)
        out_gp = self.gp.scal(a, alpha).copy_to_host()

        assert(np.allclose(out_np, out_gp))
