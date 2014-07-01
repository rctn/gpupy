from gpupy import Gpupy
import numpy as np

class test_ones():

	def setup(self):
		self.gp = Gpupy()

	def test_ones(self):
                out_np = np.ones(shape=(129,1025), dtype=np.float32)
		out_gp = self.gp.ones(shape=(129,1025))
		assert(np.allclose(out_np, out_gp))

                out_np = np.ones(shape=129, dtype=np.float32)
		out_gp = self.gp.ones(shape=129)
		assert(np.allclose(out_np, out_gp))

                out_np = np.ones(129, dtype=np.float32)
		out_gp = self.gp.ones(129)
		assert(np.allclose(out_np, out_gp))
