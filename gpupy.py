"""
gpupy.py
NumPy GPU wrapper
Jesse Livezey and Zayd Enam
May 7th 2014
"""
import numbapro.cudalib.cublas
from numbapro import cuda
import numpy as np

@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:])')
def mmultiply_pointwise(a,b,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = a[i,j]*b[i,j]

@cuda.jit('void(f4[:],f4[:],f4[:])')
def vmultiply_pointwise(a,b,out):
    n = a.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = a[i]*b[i]

class Gpupy(object):
    def __init__(self):
        self.blas = numbapro.cudalib.cublas.Blas()

    def dot(self, a, b, out=None):

        if type(a) == np.ndarray:
            a = np.array(a, order='F')

        elif type(a) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        else:
            raise NotImplementedError

        if a.dtype == np.float32:
            out_dtype = a.dtype
        else:
            raise NotImplementedError

        if type(b) == np.ndarray:
            b = np.array(b, order='F')
        elif type(b) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        else:
            raise NotImplementedError

        if b.dtype == np.float32:
            pass
        else:
            raise NotImplementedError

        if out == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError

        if b.dtype == np.float32:
            pass
        else:
            raise NotImplementedError

        a_dim = a.shape
        b_dim = b.shape

        if a.ndim == 2 and b.ndim == 2:
            if a_dim[1] != b_dim[0]:
                raise ValueError('matrices are not aligned')

            if out is None:
                out = cuda.device_array((a_dim[0], b_dim[1]), dtype=out_dtype, order='F')
            elif out.shape[0] == a_dim[0] and out.shape[1] == b_dim[1]:
                pass
            else:
                raise ValueError('matrices are not aligned')

            self.blas.gemm('N', 'N', a_dim[0], b_dim[1], a_dim[1], 1, a, b, 0, out)

        elif a.ndim == 2 and b.ndim == 1:
            if a_dim[1] != b_dim[0]:
                raise ValueError('matrices are not aligned')
            
            if out is None:
                out = cuda.device_array((a_dim[0]), dtype=out_dtype, order='F')
            elif out.shape[0] == a_dim[0]:
                pass
            else:
                raise ValueError('matrices are not aligned')

            self.blas.gemv('N', a_dim[0], a_dim[1], 1., a, b, 0., out)

        elif a.ndim == 1 and b.ndim == 2:
            if a_dim[0] != b_dim[0]:
                raise ValueError('matrices are not aligned')
            
            if out is None:
                out = cuda.device_array((b_dim[1]), dtype=out_dtype, order='F')
            elif out.shape[0] == b_dim[1]:
                pass
            else:
                raise ValueError('matrices are not aligned')

            self.blas.gemv('T', b_dim[0], b_dim[1], 1., b, a, 0., out)
        elif a.ndim == 1 and b.ndim == 1:
            if a_dim[0] != b_dim[0]:
                raise ValueError('matricies not aligned')
            out = self.blas.dot(a,b)
        else:
            raise NotImplementedError

        return out


    def T(self, a, out=None):
        if type(a) == np.ndarray:
            a = np.array(a, order='F')
        elif type(a) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        else:
            raise NotImplementedError

        if a.dtype == np.float32:
            out_dtype = a.dtype
        else:
            raise NotImplementedError
            
        if out == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out == None:
            pass
        else:
            raise NotImplementedError

        a_dim = a.shape
        if a.ndim == 2:
            if out is None:
                out = cuda.device_array((a_dim[1],a_dim[0]),dtype=out_dtype,order='F')
            elif out.shape[0] == a_dim[1] and out.shape[1] == a_dim[0]:
                pass
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.blas.geam('T','T',a_dim[1],a_dim[0],1.,a,0.,a,out)

        return out

    def add(self, a, b, out = None, alpha = 1., beta = 1.):

        if type(a) == np.ndarray:
            a = np.array(a, order='F')

        elif type(a) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        else:
            raise NotImplementedError

        if a.dtype == np.float32:
            out_dtype = a.dtype
        else:
            raise NotImplementedError

        if type(b) == np.ndarray:
            b = np.array(b, order='F')
        elif type(b) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        else:
            raise NotImplementedError

        if b.dtype == np.float32:
            pass
        else:
            raise NotImplementedError

        if out == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError

        if b.dtype == np.float32:
            pass
        else:
            raise NotImplementedError

        a_dim = a.shape
        b_dim = b.shape

        if a.ndim == 2 and b.ndim == 2:
            if a_dim[0] != b_dim[0] and a_dim[1] != b_dim[1]:
                raise ValueError('matrices are not aligned')

            if out is None:
                out = cuda.device_array((a_dim[0], b_dim[1]), dtype=out_dtype, order='F')
            elif out.shape[0] == a_dim[0] and out.shape[1] == b_dim[1]:
                pass
            else:
                raise ValueError('matrices are not aligned')

            self.blas.geam('N', 'N', a_dim[0], a_dim[1], alpha, a, beta, b, out)
        elif a.ndim == 1 and b.ndim == 1:
            #Maybe best to write a kernel for this? axpy overwrites y
            raise NotImplementedError
            if a_dim[0] != b_dim[0]:
                raise ValueError('matricies not aligned')
            out = self.blas.dot(a,b)
        else:
            raise NotImplementedError

        return out

    def mult(self,a,b,out=None):

        if type(a) == np.ndarray:
            a = np.array(a, order='F')
        elif type(a) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        else:
            raise NotImplementedError

        if a.dtype == np.float32:
            out_dtype = a.dtype
        else:
            raise NotImplementedError

        if type(b) == np.ndarray:
            b = np.array(b, order='F')
        elif type(b) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        else:
            raise NotImplementedError

        if b.dtype == np.float32:
            pass
        else:
            raise NotImplementedError

        if out == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError

        if b.dtype == np.float32:
            pass
        else:
            raise NotImplementedError

        a_dim = a.shape
        b_dim = b.shape

        if a.ndim == 2 and b.ndim == 2:
            if a_dim[0] != b_dim[0] and a_dim[1] != b_dim[1]:
                raise ValueError('matrices are not aligned')

            if out is None:
                out = cuda.device_array((a_dim[0], a_dim[1]), dtype=out_dtype, order='F')
            elif out.shape[0] == a_dim[0] and out.shape[1] == a_dim[1]:
                pass
            else:
                raise ValueError('matrices are not aligned')

            blockdim2 = (32,32)
            griddim2 = (int(a_dim[0]/blockdim2[0]),int(a_dim[1]/blockdim2[1]))
            mmultiply_pointwise[griddim2,blockdim2](a,b,out)

        elif a.ndim == 1 and b.ndim == 1:
            if a_dim[0] != b_dim[0]:
                raise ValueError('matricies not aligned')
            if out is None:
                out = cuda.device_array(a_dim[0], dtype=out_dtype, order='F')
            elif out.shape[0] == a_dim[0]:
                pass
            else:
                raise ValueError('matrices are not aligned')
            blockdim = 32
            griddim = int(a_dim[0]/blockdim)
            vmultiply_pointwise[griddim,blockdim](a,b,out)
        else:
            raise NotImplementedError

        return out
