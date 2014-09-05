from gpupy.gpupy import Gpupy
import numpy as np

class test_reshape():

    def setup(self):
        self.gp = Gpupy()

    def test_reshape(self):
        """ Reshapes array.
        """
        a = np.array(np.arange(10), dtype=np.float32, order='F')

        out_np = np.reshape(a, newshape=(2,5), order='F')
        out_gp = self.gp.reshape(a, newshape=(2,5)).copy_to_host()
        assert(np.allclose(out_np, out_gp))

        a = np.array(np.arange(10), dtype=np.float32, order='c')

        out_np = np.reshape(a, newshape=(2,5), order='C')
        out_gp = self.gp.reshape(a, newshape=(2,5), order='C').copy_to_host()
        assert(np.allclose(out_np, out_gp))
