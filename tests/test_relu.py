from gpupy import Gpupy
import numpy as np
import copy

class test_relu():

	def setup(self):
            self.gp = Gpupy()
            self.rng = np.random.RandomState(0)

	def test_relu_m(self):
            """relu on matrices."""
            a = self.rng.rand(129, 1025).astype(np.float32)
            t = .5

            out_np = copy.deepcopy(a)
            out_np[out_np<t] = 0.
            out_gp = self.gp.relu(a, thresh=t).copy_to_host()

            assert(np.allclose(out_np, out_gp))

            a = self.rng.rand(129, 1025).astype(np.float32)
            val = .5
            out_np = copy.deepcopy(a)
            out_np[out_np<t] = val
            out_gp = self.gp.relu(a, thresh=t, set_val=val).copy_to_host()

            assert(np.allclose(out_np, out_gp))

	def test_relu_v(self):
            """relu on vectors"""
            a = self.rng.rand(129).astype(np.float32)
            t = .5

            out_np = copy.deepcopy(a)
            out_np[out_np<t] = 0.
            out_gp = self.gp.relu(a, thresh=t).copy_to_host()

            assert(np.allclose(out_np, out_gp))

            a = self.rng.rand(129).astype(np.float32)
            val = .5
            out_np = copy.deepcopy(a)
            out_np[out_np<t] = val
            out_gp = self.gp.relu(a, thresh=t, set_val=val).copy_to_host()

            assert(np.allclose(out_np, out_gp))
