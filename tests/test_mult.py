from gpupy import Gpupy
import numpy as np

class test_mult():

    def setup(self):
        self.gp = Gpupy()
        self.rng = np.random.RandomState(0)

    def test_matrix_multiply(self):
        a = self.rng.rand(129, 1025).astype(np.float32)
        b = self.rng.rand(129, 1025).astype(np.float32)

        out_np = a*b
        out_gp = self.gp.mult(a,b).copy_to_host()

        assert(np.allclose(out_np, out_gp))

    def test_vector_multiply(self):
        a = self.rng.rand(1025).astype(np.float32)
        b = self.rng.rand(1025).astype(np.float32)

        out_np = a*b
        out_gp = self.gp.mult(a,b).copy_to_host()

        assert(np.allclose(out_np, out_gp))
