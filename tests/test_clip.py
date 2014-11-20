from gpupy.gpupy import Gpupy
import numpy as np
import copy
from numbapro import cuda

class test_relu():

    def setup(self):
        self.gp = Gpupy()
        self.rng = np.random.RandomState(0)

    def test_clip_m(self):
        """clip on matrices."""
        a = self.rng.rand(129, 1025).astype(np.float32)
        out_np = np.clip(a,-.1,.1)
        out_gp = self.gp.clip(a,-.1,.1).copy_to_host()
        assert(np.allclose(out_np, out_gp))

        a = self.rng.rand(129, 1025).astype(np.float32)
        out_np = np.clip(a,-.1,.1)
        out_gp = cuda.to_device(a)
        self.gp.clip(a, -.1, .1, out=out_gp)
        assert(np.allclose(out_np, out_gp.copy_to_host()))

    def test_clip_v(self):
        """clip on vectors."""
        a = self.rng.rand(1025).astype(np.float32)
        out_np = np.clip(a,-.1,.1)
        out_gp = self.gp.clip(a,-.1,.1).copy_to_host()
        assert(np.allclose(out_np, out_gp))

        a = self.rng.rand(1025).astype(np.float32)
        out_np = np.clip(a,-.1,.1)
        out_gp = cuda.to_device(a)
        self.gp.clip(a, -.1, .1, out=out_gp)
        assert(np.allclose(out_np, out_gp.copy_to_host()))
