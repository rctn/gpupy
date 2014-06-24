from gpupy import Gpupy
import numpy as np

class test_T():

	def setup(self):
		self.gp = Gpupy()
		self.rng = np.random.RandomState(0)

	def test_transpose(self):
		a = self.rng.rand(129, 1025).astype(np.float32)

		out_np = a.T
		out_gp = self.gp.T(a).copy_to_host()

		assert(np.allclose(out_np, out_gp))

