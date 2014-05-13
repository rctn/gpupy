from gpupy import Gpupy
import numpy as np

class test_dot():

	def setup(self):
		self.gp = Gpupy()
		self.rng = np.random.RandomState(0)

	def test_gemm(self):
		a = self.rng.rand(128, 1024).astype(np.float32)
		b = self.rng.rand(1024, 256).astype(np.float32)

		out_np = np.dot(a, b)
		out_gp = self.gp.dot(a, b).copy_to_host()

		assert(np.allclose(out_np, out_gp))

	def test_gemv(self):		
		a = self.rng.rand(128, 1024).astype(np.float32)
		b = self.rng.rand(1024).astype(np.float32)

		out_np = np.dot(a, b)
		out_gp = self.gp.dot(a, b).copy_to_host()

		assert(np.allclose(out_np, out_gp))

	def test_gevm(self):		
		a = self.rng.rand(128, 1024).astype(np.float32)
		b = self.rng.rand(128).astype(np.float32)

		out_np = np.dot(b,a)
		out_gp = self.gp.dot(b,a).copy_to_host()

		assert(np.allclose(out_np, out_gp))

	def test_vvdot(self):		
		a = self.rng.rand(128).astype(np.float32)
		b = self.rng.rand(128).astype(np.float32)

		out_np = np.dot(a,b)
		out_gp = self.gp.dot(a,b)

		assert(np.allclose(out_np, out_gp))
