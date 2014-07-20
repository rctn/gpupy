from gpupy import Gpupy
import numpy as np

class test_sum():

	def setup(self):
            self.gp = Gpupy()
            self.rng = np.random.RandomState(0)

	def test_sum(self):
            """Test sum function."""
            a = self.rng.rand(129, 1025).astype(np.float32)

            out_np = a.sum()
            out_gp = self.gp.sum(a)

            assert(np.allclose(out_np, out_gp))

            a = self.rng.rand(129).astype(np.float32)

            out_np = a.sum()
            out_gp = self.gp.sum(a)

            assert(np.allclose(out_np, out_gp))

	def test_sum(self):
            """Test sum function along axes."""
            a = self.rng.rand(1, 33).astype(np.float32)

            out_np = a.sum(axis=0).astype(np.float32)
            out_gp = self.gp.sum(a, axis=0).copy_to_host()
            print out_np
            print out_gp

            assert(np.allclose(out_np, out_gp))

            out_np = a.sum(axis=1)
            out_gp = self.gp.sum(a, axis=1).copy_to_host()

            assert(np.allclose(out_np, out_gp))

