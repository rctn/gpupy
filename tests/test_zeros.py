from gpupy import Gpupy
import numpy as np

class test_zeros():

	def setup(self):
		self.gp = Gpupy()

	def test_zeros(self):
                out_np = np.zeros(shape=(129,1025), dtype=np.float32)
		out_gp = self.gp.zeros(shape=(129,1025)).copy_to_host()
		assert(np.allclose(out_np, out_gp))

                out_np = np.zeros(shape=129, dtype=np.float32)
		out_gp = self.gp.zeros(shape=129).copy_to_host()
		assert(np.allclose(out_np, out_gp))

                out_np = np.zeros(129, dtype=np.float32)
		out_gp = self.gp.zeros(129).copy_to_host()
		assert(np.allclose(out_np, out_gp))
