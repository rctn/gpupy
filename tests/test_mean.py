from gpupy import Gpupy
import numpy as np

class test_mean():

	def setup(self):
            self.gp = Gpupy()
            self.rng = np.random.RandomState(0)

	def test_mean(self):
            """Test mean function."""
            a = self.rng.rand(129, 1025).astype(np.float32)

            out_np = a.mean()
            out_gp = self.gp.mean(a)

            assert(np.allclose(out_np, out_gp))

            out_np = a.mean(axis=0)
            out_gp = self.gp.mean(a, axis=0).copy_to_host()

            assert(np.allclose(out_np, out_gp))

            out_np = a.mean(axis=1)
            out_gp = self.gp.mean(a, axis=1).copy_to_host()

            assert(np.allclose(out_np, out_gp))

            a = self.rng.rand(1025).astype(np.float32)

            out_np = a.mean()
            out_gp = self.gp.mean(a)

            assert(np.allclose(out_np, out_gp))
