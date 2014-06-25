from gpupy import Gpupy
import numpy as np

class test_diag():

	def setup(self):
		self.gp = Gpupy()
		self.rng = np.random.RandomState(0)

	def test_v2m(self):
		a = self.rng.rand(129).astype(np.float32)

		out_np = np.diag(a)
		out_gp = self.gp.diag(a).copy_to_host()
		assert(np.allclose(out_np, out_gp))

	def test_m2v(self):
		a = self.rng.rand(129, 129).astype(np.float32)

		out_np = np.diag(a)
		out_gp = self.gp.diag(a).copy_to_host()
		assert(np.allclose(out_np, out_gp))
