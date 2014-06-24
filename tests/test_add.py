from gpupy import Gpupy
import numpy as np

class test_add():

    def setup(self):
        self.gp = Gpupy()
        self.rng = np.random.RandomState(0)

    def test_madd(self):
        a = self.rng.rand(129, 1025).astype(np.float32)
        b = self.rng.rand(129, 1025).astype(np.float32)

        out_np = a+b
        out_gp = self.gp.add(a,b).copy_to_host()

        assert(np.allclose(out_np, out_gp))

    def test_madd_scale(self):
        a = self.rng.rand(129, 1025).astype(np.float32)
        b = self.rng.rand(129, 1025).astype(np.float32)
        alpha = .4
        beta = -1.6

        out_np = (alpha*a+beta*b).astype(np.float32)
        out_gp = self.gp.add(a,b,alpha = alpha, beta = beta).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))

    def test_vadd(self):
        a = self.rng.rand(1025).astype(np.float32)
        b = self.rng.rand(1025).astype(np.float32)

        out_np = a+b
        out_gp = self.gp.add(a,b).copy_to_host()

        assert(np.allclose(out_np, out_gp))

    def test_vadd_scale(self):
        a = self.rng.rand(1025).astype(np.float32)
        b = self.rng.rand(1025).astype(np.float32)
        alpha = .4
        beta = -1.6

        out_np = (alpha*a+beta*b).astype(np.float32)
        out_gp = self.gp.add(a,b,alpha = alpha, beta = beta).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))

    def test_ms_scale(self):
        a = self.rng.rand(129,1025).astype(np.float32)
        b = 1.
        alpha = .4
        beta = -1.6

        out_np = (alpha*a+beta*b).astype(np.float32)
        out_gp = self.gp.add(a,b,alpha = alpha, beta = beta).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))

    def test_ms(self):
        a = self.rng.rand(129,1025).astype(np.float32)
        b = 1.

        out_np = (a+b).astype(np.float32)
        out_gp = self.gp.add(a,b).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))

    def test_vs_scale(self):
        a = self.rng.rand(1025).astype(np.float32)
        b = 1.
        alpha = .4
        beta = -1.6

        out_np = (alpha*a+beta*b).astype(np.float32)
        out_gp = self.gp.add(a,b,alpha = alpha, beta = beta).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))

    def test_vs(self):
        a = self.rng.rand(1025).astype(np.float32)
        b = 1.

        out_np = (a+b).astype(np.float32)
        out_gp = self.gp.add(a,b).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))

    def test_mv_scale(self):
        a = self.rng.rand(129,1025).astype(np.float32)
        b = self.rng.rand(1025).astype(np.float32)
        alpha = .4
        beta = -1.6

        out_np = (alpha*a+beta*b).astype(np.float32)
        out_gp = self.gp.add(a,b,alpha = alpha, beta = beta).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))

    def test_mv(self):
        a = self.rng.rand(129,1025).astype(np.float32)
        b = self.rng.rand(1025).astype(np.float32)

        out_np = (a+b).astype(np.float32)
        out_gp = self.gp.add(a,b).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))

    def test_ma_scale(self):
        a = self.rng.rand(1025).astype(np.float32)[:,np.newaxis]
        b = self.rng.rand(1025).astype(np.float32)
        alpha = .4
        beta = -1.6

        out_np = (alpha*a+beta*b).astype(np.float32)
        out_gp = self.gp.add(a,b,alpha = alpha, beta = beta).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))

    def test_ma(self):
        a = self.rng.rand(1025).astype(np.float32)[:,np.newaxis]
        b = self.rng.rand(1025).astype(np.float32)

        out_np = (a+b).astype(np.float32)
        out_gp = self.gp.add(a,b).copy_to_host()
        print out_np
        print out_gp

        assert(np.allclose(out_np, out_gp, atol = 1.e-5))
