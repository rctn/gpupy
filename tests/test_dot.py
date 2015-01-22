from gpupy import Gpupy
import numpy as np

class test_dot():

    def setup(self):
        self.gp = Gpupy()
        self.rng = np.random.RandomState(0)

    def test_gemm(self):
        """ Test matrix-matrix dot product."""
        a = self.rng.rand(129, 1025).astype(np.float32)
        b = self.rng.rand(1025, 257).astype(np.float32)

        out_np = np.dot(a, b)
        out_gp = self.gp.dot(a, b).copy_to_host()
        assert(np.allclose(out_np, out_gp))

        out_gp = self.gp.zeros(shape=(129, 257))
        self.gp.dot(a, b, out=out_gp)
        assert(np.allclose(out_np, out_gp.copy_to_host()))

    def test_gemv(self):		
        a = self.rng.rand(129, 1025).astype(np.float32)
        b = self.rng.rand(1025).astype(np.float32)

        out_np = np.dot(a, b)
        out_gp = self.gp.dot(a, b).copy_to_host()
        assert(np.allclose(out_np, out_gp))

        out_gp = self.gp.zeros(shape=129)
        self.gp.dot(a, b, out=out_gp)
        assert(np.allclose(out_np, out_gp.copy_to_host()))

    def test_gevm(self):		
        a = self.rng.rand(129, 1025).astype(np.float32)
        b = self.rng.rand(129).astype(np.float32)

        out_np = np.dot(b,a)
        out_gp = self.gp.dot(b,a).copy_to_host()
        assert(np.allclose(out_np, out_gp))

        out_gp = self.gp.zeros(shape=1025)
        self.gp.dot(b, a, out=out_gp)
        assert(np.allclose(out_np, out_gp.copy_to_host()))

    def test_vvdot(self):		
        a = self.rng.rand(129).astype(np.float32)
        b = self.rng.rand(129).astype(np.float32)

        out_np = np.dot(a,b)
        out_gp = self.gp.dot(a,b)
        assert(np.allclose(out_np, out_gp))
