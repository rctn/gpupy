""" Numpy-like GPU wrapper. Contains the main gpupy class 
and supporting numbapro cuda kernels.

Jesse Livezey and Zayd Enam
May 7th 2014
"""
from __future__ import division
__authors__   = "Jesse Livezey, Zayd Enam"
__copyright__ = "(c) 2014, Jesse Livezey, Zayd Enam"
__license__   = "The MIT License (MIT)"
__contact__   = "Jesse Livezey <jesse.livezey@berkeley.edu>"
import numbapro.cudalib.cublas
from numbapro import cuda
import numpy as np
from math import ceil, exp, fabs, log, tanh

class Gpupy(object):
    """Class that has cuBLAS wrappers and additional GPU functionality.
    
    Behaves like numpy functions whenever possible.
    
    Parameters
    ----------
    gpuID : int
        ID of GPU to use. Cannot be changed later and must be set before any
        cuda functionality is used.
    """

    def __init__(self, gpuID=None, stream=None):
        if gpuID is not None:
            if gpuID < len(cuda.list_devices()) and gpuID >= 0:
                cuda.close()
                cuda.select_device(gpuID)
            else:
                raise ValueError('GPU ID not found')
        if stream is None:
            self.stream = cuda.stream()
        else:
            assert isinstance(stream, numba.cuda.cudadrv.driver.Stream)
            self.stream = stream
        self.blas = numbapro.cudalib.cublas.Blas(stream=self.stream)
        self.blockdim = 32
        self.blockdim2 = (32, 32)

    def synchronize(self):
        """Synchronize cuda stream."""
        self.stream.synchronize()

    def sync(self):
        """Alias to synchronize."""
        self.synchronize()

    def array(self, a):
        """Creates an on-GPU array from a, a numpy array.
        
        Parameters
        ----------
        a : array-like
            Array to be send to GPU.
        """
        if type(a) != np.ndarray:
            raise NotImplementedError

        out, out_dtype = _check_array(a)
        return out

    def dot(self, a, b, out=None):
        """Takes the dot product of two 2D arrays or 1D vectors.

        Checks array type and shape. Should behave like numpy.dot(a, b).

        Parameters
        ----------
        a : array-like
            Numpy or DeviceNDArray
        b : array-like
            Numpy or DeviceNDArray
        out : DeviceNDArray (optional)
            Array will be filled with result if given.
        """

        b, out_dtype = _check_array(b)
        a, out_dtype = _check_array(a)

        if isinstance(out, cuda.cudadrv.devicearray.DeviceNDArray):
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

            self.blas.gemm('N', 'N', a_dim[0], b_dim[1], a_dim[1], 1., a, b, 0., out)

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
        """Returns the transpose of a 2D array.

        Parameters
        ----------
        a : array-like
            Numpy or DeviceNDArray to transpose.
        out : DeviceNDArray (optional)
            Array to overwrite with result.
        """

        a, out_dtype = _check_array(a)
            
        if type(out) == cuda.cudadrv.devicearray.DeviceNDArray:
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

    def transpose(self, a, out=None):
        """Alias to T."""
        return self.T(a, out)

    def add(self, a, b, out = None, alpha = 1., beta = 1.):
        """Pointwise addition of two scalars, 1D, or 2D arrays.

        Behaves like numpy array in terms of broadcasting.

        Parameters
        ----------
        a : array-like
            Array to add.
        b : array-like
            Array to add.
        out : DeviceNDArray (optional)
            Result will overwrite out if given.
        alpha : float (optional)
            Scales a before addition.
        beta : float
            Scales b before addition.
        """

        b, out_dtype = _check_array(b)
        a, out_dtype = _check_array(a)

        if type(out) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError

        a_dim = a.shape
        b_dim = b.shape

        # Matrix-matrix addition
        if a.ndim == 2 and b.ndim == 2:
            # Full-size matricies
            if a_dim == b_dim:
                if out is None:
                    out = cuda.device_array((a_dim[0], a_dim[1]), dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0] and out.shape[1] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')

                self.blas.geam('N', 'N', a_dim[0], a_dim[1], alpha, a, beta, b, out)
            # np.newaxis matrices
            elif a_dim[0] == b_dim[0] and b_dim[1] == 1:
                if out is None:
                    out = cuda.device_array((a_dim[0], a_dim[1]), dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0] and out.shape[1] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                blockdim = (32,32)
                griddim = (int(ceil(a_dim[0]/blockdim[0])),int(ceil(a_dim[1]/blockdim[1])))
                if alpha != 1. or beta != 1.:
                    m_mn_sadd_pointwise[griddim,blockdim, self.stream](a,b,alpha,beta,out)
                else:
                    m_mn_add_pointwise[griddim,blockdim, self.stream](a,b,out)
            elif a_dim[1] == b_dim[1] and b_dim[0] == 1:
                if out is None:
                    out = cuda.device_array((a_dim[0], a_dim[1]), dtype=out_dtype, order='F')
                elif out.shape == a_dim:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                blockdim = (32,32)
                griddim = (int(ceil(a_dim[0]/blockdim[0])),int(ceil(a_dim[1]/blockdim[1])))
                if alpha != 1. or beta != 1.:
                    m_nm_sadd_pointwise[griddim,blockdim, self.stream](a,b,alpha,beta,out)
                else:
                    m_nm_add_pointwise[griddim,blockdim, self.stream](a,b,out)
            elif b_dim[0] == a_dim[0] and a_dim[1] == 1:
                if out is None:
                    out = cuda.device_array((b_dim[0], b_dim[1]), dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0] and out.shape[1] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                blockdim = (32,32)
                griddim = (int(ceil(b_dim[0]/blockdim[0])),int(ceil(b_dim[1]/blockdim[1])))
                if alpha != 1. or beta != 1.:
                    m_mn_sadd_pointwise[griddim,blockdim, self.stream](b,a,beta,alpha,out)
                else:
                    m_mn_add_pointwise[griddim,blockdim, self.stream](b,a,out)
            elif b_dim[1] == a_dim[1] and a_dim[0] == 1:
                if out is None:
                    out = cuda.device_array((b_dim[0], b_dim[1]), dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0] and out.shape[1] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                blockdim = (32,32)
                griddim = (int(ceil(b_dim[0]/blockdim[0])),int(ceil(b_dim[1]/blockdim[1])))
                if alpha != 1. or beta != 1.:
                    m_nm_sadd_pointwise[griddim,blockdim, self.stream](b,a,beta,alpha,out)
                else:
                    m_nm_add_pointwise[griddim,blockdim, self.stream](b,a,out)
            else:
                raise ValueError('matrices are not aligned')
        # Vector-vector addition
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
            griddim = int(ceil(a_dim[0]/blockdim))
            if alpha != 1. or beta != 1.:
                vsadd_pointwise[griddim,blockdim, self.stream](a,b,alpha,beta,out)
            else:
                vadd_pointwise[griddim,blockdim, self.stream](a,b,out)
        # Matrix-scalar addition
        elif a.ndim == 2 and b.ndim == 0:
            if out is None:
                out = cuda.device_array(a_dim, dtype=out_dtype, order='F')
            elif out.shape == a_dim:
                pass
            else:
                raise ValueError('matrices are not aligned')
            blockdim = (32,32)
            griddim = (int(ceil(a_dim[0]/blockdim[0])),int(ceil(a_dim[1]/blockdim[0])))
            if alpha != 1. or beta != 1.:
                ms_sadd_pointwise[griddim,blockdim, self.stream](a,b,alpha,beta,out)
            else:
                ms_add_pointwise[griddim,blockdim, self.stream](a,b,out)
        # Scalar-matrix addition
        elif a.ndim == 0 and b.ndim == 2:
            if out is None:
                out = cuda.device_array(b_dim, dtype=out_dtype, order='F')
            elif out.shape == b_dim:
                pass
            else:
                raise ValueError('matrices are not aligned')
            blockdim = (32,32)
            griddim = (int(ceil(b_dim[0]/blockdim[0])),int(ceil(b_dim[1]/blockdim[0])))
            if alpha != 1. or beta != 1.:
                ms_sadd_pointwise[griddim,blockdim, self.stream](b,a,beta,alpha,out)
            else:
                ms_add_pointwise[griddim,blockdim, self.stream](b,a,out)
        # Vector-scalar addition
        elif a.ndim == 1 and b.ndim == 0:
            if out is None:
                out = cuda.device_array(a_dim, dtype=out_dtype, order='F')
            elif out.shape == a_dim:
                pass
            else:
                raise ValueError('matrices are not aligned')
            blockdim = 32
            griddim = int(ceil(a_dim[0]/blockdim))
            if alpha != 1. or beta != 1.:
                vs_sadd_pointwise[griddim,blockdim, self.stream](a,b,alpha,beta,out)
            else:
                vs_add_pointwise[griddim,blockdim, self.stream](a,b,out)
        # Scalar-vector addition
        elif a.ndim == 0 and b.ndim == 1:
            if out is None:
                out = cuda.device_array(b_dim, dtype=out_dtype, order='F')
            elif out.shape == b_dim:
                pass
            else:
                raise ValueError('matrices are not aligned')
            blockdim = 32
            griddim = int(ceil(b_dim[0]/blockdim))
            if alpha != 1. or beta != 1.:
                vs_sadd_pointwise[griddim,blockdim, self.stream](b,a,beta,alpha,out)
            else:
                vs_add_pointwise[griddim,blockdim, self.stream](b,a,out)
        # Matrix-vector addition
        elif a.ndim == 2 and b.ndim == 1:
            if out is None:
                out = cuda.device_array(a_dim, dtype=out_dtype, order='F')
            elif out.shape == a_dim:
                pass
            else:
                raise ValueError('matrices are not aligned')
            blockdim = (32,32)
            griddim = (int(ceil(a_dim[0]/blockdim[0])),int(ceil(a_dim[1]/blockdim[0])))
            if b.shape[0] == a.shape[0]:
                if alpha != 1. or beta != 1.:
                    mv0_sadd_pointwise[griddim,blockdim, self.stream](a,b,alpha,beta,out)
                else:
                    mv0_add_pointwise[griddim,blockdim, self.stream](a,b,out)
            elif b.shape[0] == a.shape[1]:
                if alpha != 1. or beta != 1.:
                    mv1_sadd_pointwise[griddim,blockdim, self.stream](a,b,alpha,beta,out)
                else:
                    mv1_add_pointwise[griddim,blockdim, self.stream](a,b,out)
            else:
                raise ValueError('matricies are not aligned')
        # Vector-matrix addition
        elif a.ndim == 1 and b.ndim == 2:
            if out is None:
                out = cuda.device_array(b_dim, dtype=out_dtype, order='F')
            elif out.shape == b_dim:
                pass
            else:
                raise ValueError('matrices are not aligned')
            blockdim = (32,32)
            griddim = (int(ceil(b_dim[0]/blockdim[0])),int(ceil(b_dim[1]/blockdim[0])))
            if a.shape[0] == b.shape[0]:
                if alpha != 1. or beta != 1.:
                    mv0_sadd_pointwise[griddim,blockdim, self.stream](b,a,beta,alpha,out)
                else:
                    mv0_add_pointwise[griddim,blockdim, self.stream](b,a,out)
            elif a.shape[0] == b.shape[1]:
                if alpha != 1. or beta != 1.:
                    mv1_sadd_pointwise[griddim,blockdim, self.stream](b,a,beta,alpha,out)
                else:
                    mv1_add_pointwise[griddim,blockdim, self.stream](b,a,out)
            else:
                raise ValueError('matricies are not aligned')
        else:
            raise NotImplementedError
        return out

    def multiply(self, a, b, out=None):
        "Alias to mult."
        return mult(a, b, out)

    def mult(self, a, b, out=None, alpha=None):
        """Pointwise multiplication of two 1D or 2D arrays.

        Parameters
        ----------
        a : array-like
            Array to multiply.
        b : array-like
            Array to multiply.
        out : DeviceNDArray (optional)
            Result will overwrite out if given.
        alpha : float
            Additional scale factor for multiplication.
        """

        if alpha is not None:
            raise NotImplementedError

        b, out_dtype = _check_array(b)
        a, out_dtype = _check_array(a)

        if type(out) == cuda.cudadrv.devicearray.DeviceNDArray:
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
            griddim2 = (int(ceil(a_dim[0]/blockdim2[0])),int(ceil(a_dim[1]/blockdim2[1])))
            mmultiply_pointwise[griddim2,blockdim2, self.stream](a,b,out)

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
            griddim = int(ceil(a_dim[0]/blockdim))
            vmultiply_pointwise[griddim,blockdim, self.stream](a,b,out)
        else:
            raise NotImplementedError

        return out

    def sub(self, a, b, out = None, alpha = 1., beta = 1.):
        return  self.add(a, b, out=out, alpha=alpha, beta=-beta)

    def scal(self, a, alpha):
        """Scale a 1D or 2D array by alpha.

        Parameters
        ----------
        a : array-like
            Array to scale.
        alpha : float
            Scaling factor.
        """

        a, out_dtype = _check_array(a)

        a_dim = a.shape

        if a.ndim == 2:
            a_strides = a.strides
            a_dtype = a.dtype
            d_flat_a = _cu_reshape(a, (np.prod(a_dim),), (a_strides[0],), a_dtype)
            self.blas.scal(alpha, d_flat_a)
            a = _cu_reshape(d_flat_a, a_dim, a_strides, a_dtype)
        elif a.ndim == 1:
            if type(a) == np.ndarray:
                a = cuda.to_device(a)
            self.blas.scal(alpha, a)
        else:
            raise NotImplementedError

        return a

    def sum(self, a, out=None, axis=None):
        """Sum array elements.

        Parameters
        ----------
        a : array-like
           Array to sum.
        out : array-like
            Result will be stored in this array.
        axis : int
            1 or 0 for 2D arrays.
        """
        a, out_dtype = _check_array(a)

        a_dim = a.shape
        
        if a.ndim == 2:
            if axis is None:
                a_strides = a.strides
                d_flat_a = _cu_reshape(a, (np.prod(a_dim),), (a_strides[0],), out_dtype)
                out = self.blas.asum(d_flat_a)
            elif axis == 0:
                if out is None:
                    out = cuda.device_array(a_dim[1], dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                griddim = int(ceil(a_dim[1]/self.blockdim))
                sum_0[griddim, self.blockdim, self.stream](a, out)
            elif axis == 1:
                if out is None:
                    out = cuda.device_array(a_dim[0], dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                griddim = int(ceil(a_dim[0]/self.blockdim))
                sum_1[griddim, self.blockdim, self.stream](a, out)
        elif a.ndim == 1:
            out = self.blas.asum(a)
            pass
        else:
            raise NotImplementedError
        return out

    def mean(self, a, out=None, axis=None):
        """Average array elements.

        Parameters
        ----------
        a : array-like
           Array to average.
        out : array-like
            Result will be stored in this array.
        axis : int
            1 or 0 for 2D arrays.
        """
        a, out_dtype = _check_array(a)

        a_dim = a.shape
        
        if a.ndim == 2:
            if axis is None:
                a_strides = a.strides
                d_flat_a = _cu_reshape(a, (np.prod(a_dim),), (a_strides[0],), out_dtype)
                out = self.blas.asum(d_flat_a)/float(np.prod(a_dim))
            elif axis == 0:
                if out is None:
                    out = cuda.device_array(a_dim[1], dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                griddim = int(ceil(a_dim[1]/self.blockdim))
                mean_0[griddim, self.blockdim, self.stream](a, float(a_dim[0]), out)
            elif axis == 1:
                if out is None:
                    out = cuda.device_array(a_dim[0], dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                griddim = int(ceil(a_dim[0]/self.blockdim))
                mean_1[griddim, self.blockdim, self.stream](a, float(a_dim[1]), out)
        elif a.ndim == 1:
            out = self.blas.asum(a)/float(np.prod(a_dim))
            pass
        else:
            raise NotImplementedError
        return out

    def diag(self, a, out=None):
        """Creates vector from diagonal of matrix or
        matrix with diagonal from vector.

        Parameters
        ----------
        a : array-like
            Vector or array from which to take diagonal.
        out : array-like, optional
            Output array.
        """
        a, out_dtype = _check_array(a)

        a_dim = a.shape

        if a.ndim == 2:
            if out is None:
                out = cuda.device_array(shape=a_dim[0], dtype=out_dtype, order='F')
            elif out.shape[0] == a_dim[0] and out.ndim == 1:
                pass
            else:
                raise ValueError('matrices are not aligned')
            griddim = int(ceil(a_dim[0]/self.blockdim))
            diag2v[griddim, self.blockdim, self.stream](a, out)

        elif a.ndim == 1:
            if out is None:
                out = cuda.device_array(shape=(a_dim[0],a_dim[0]), dtype=out_dtype, order='F')
            elif out.shape == (a_dim[0], a_dim[0]):
                pass
            else:
                raise ValueError('matrices are not aligned')
            griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])), int(ceil(a_dim[0]/self.blockdim2[1])))
            diag2m[griddim2, self.blockdim2, self.stream](a, out)
        else:
            raise NotImplementedError
        
        return out

    def zero_diag(self, a):
        """Set diagonal of matrix to zero.

        Parameters
        ----------
        a : array-like
            Array to set diagonal to zero
        """
        a, out_dtype = _check_array(a)

        a_dim = a.shape

        if a.ndim == 2:
            griddim = int(ceil(a_dim[0]/self.blockdim))
            zero_diag_m[griddim, self.blockdim, self.stream](a)
        else:
            raise NotImplementedError

        return a

    def set_diag(self, a, value):
        """Set diagonal of matrix to value.

        Parameters
        ----------
        a : array-like
            Array to set diagonal to value.
        value : array-like
            Value or array of values for diagonal.
        """
        value, out_dtype = _check_array(value)
        a, out_dtype = _check_array(a)

        a_dim = a.shape
        if a_dim[0] != a_dim[1]:
            raise NotImplementedError
        if value.ndim !=1:
            raise NotImplementedError

        if a.ndim == 2:
            if value.shape[0] == 1:
                griddim = int(ceil(a_dim[0]/self.blockdim))
                set_diag_s[griddim, self.blockdim, self.stream](a, value)
            elif value.shape[0] == a_dim[0]:
                griddim = int(ceil(a_dim[0]/self.blockdim))
                set_diag_v[griddim, self.blockdim, self.stream](a, value)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return a

    def exp(self, a, out=None):
        """Exponentiate input.

        Parameters
        ----------
        a : array-like
            Array to rectify.
        """

        a, out_dtype = _check_array(a)
        a_dim = a.shape

        if type(out) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError
        if out is None:
            out = cuda.device_array(shape=a_dim, dtype=out_dtype, order='F')
        elif out.shape == a_dim:
            pass
        else:
            raise ValueError('matrices are not aligned')

        if a.ndim == 2:
            griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
            exp_m[griddim2, self.blockdim2, self.stream](a, out)
        elif a.ndim == 1:
            griddim = int(ceil(a_dim[0]/self.blockdim))
            exp_v[griddim, self.blockdim, self.stream](a, out)
        else:
            raise NotImplementedError

        return out

    def log(self, a, out=None):
        """Exponentiate input.

        Parameters
        ----------
        a : array-like
            Array to rectify.
        """

        a, out_dtype = _check_array(a)
        a_dim = a.shape

        if type(out) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError
        if out is None:
            out = cuda.device_array(shape=a_dim, dtype=out_dtype, order='F')
        elif out.shape == a_dim:
            pass
        else:
            raise ValueError('matrices are not aligned')

        if a.ndim == 2:
            griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
            log_m[griddim2, self.blockdim2, self.stream](a, out)
        elif a.ndim == 1:
            griddim = int(ceil(a_dim[0]/self.blockdim))
            log_v[griddim, self.blockdim, self.stream](a, out)
        else:
            raise NotImplementedError

        return out

    def abs(self, a, out=None):
        """ABS of input.

        Parameters
        ----------
        a : array-like
            Array to rectify.
        """

        a, out_dtype = _check_array(a)
        a_dim = a.shape

        if type(out) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError
        if out is None:
            out = cuda.device_array(shape=a_dim, dtype=out_dtype, order='F')
        elif out.shape == a_dim:
            pass
        else:
            raise ValueError('matrices are not aligned')

        if a.ndim == 2:
            griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
            abs_m[griddim2, self.blockdim2, self.stream](a, out)
        elif a.ndim == 1:
            griddim = int(ceil(a_dim[0]/self.blockdim))
            abs_v[griddim, self.blockdim, self.stream](a, out)
        else:
            raise NotImplementedError

        return out

    def tanh(self, a, out=None):
        """Tanh of input.

        Parameters
        ----------
        a : array-like
            Array to rectify.
        """

        a, out_dtype = _check_array(a)
        a_dim = a.shape

        if type(out) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError
        if out is None:
            out = cuda.device_array(shape=a_dim, dtype=out_dtype, order='F')
        elif out.shape == a_dim:
            pass
        else:
            raise ValueError('matrices are not aligned')

        if a.ndim == 2:
            griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
            tanh_m[griddim2, self.blockdim2, self.stream](a, out)
        elif a.ndim == 1:
            griddim = int(ceil(a_dim[0]/self.blockdim))
            tanh_v[griddim, self.blockdim, self.stream](a, out)
        else:
            raise NotImplementedError

        return out

    def relu(self, a, thresh=0., set_val=0., flip_x=0, out=None):
        """Rectify input.

        Parameters
        ----------
        a : array-like
            Array to rectify.
        thresh : float, optional
            Value to start rectification.
        flip_x : int (1 or 0), optional
            Whether to rectify negative half (default) or positive half.
        """

        thresh, out_dtype = _check_array(thresh)
        thresh_dim = thresh.shape

        a, out_dtype = _check_array(a)
        a_dim = a.shape

        if type(out) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError
        if out is None:
            out = a
        elif out.shape == a_dim:
            pass
        else:
            raise ValueError('matrices are not aligned')

        if a.ndim == 2:
            if thresh.ndim == 2:
                if a_dim == thresh_dim:
                    griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
                    thresh_m_t[griddim2, self.blockdim2, self.stream](a, thresh, flip_x, set_val, out)
                elif a_dim[0] == thresh_dim[0] and thresh_dim[1] == 1:
                    griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
                    thresh_m_tn[griddim2, self.blockdim2, self.stream](a, thresh, flip_x, set_val, out)
                elif a_dim[1] == thresh_dim[1] and thresh_dim[0] == 1:
                    griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
                    thresh_m_nt[griddim2, self.blockdim2, self.stream](a, thresh, flip_x, set_val, out)
                else:
                    raise ValueError('matrices are not aligned')
            elif thresh.ndim == 1:
                if thresh_dim[0] == a_dim[0]:
                    griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
                    thresh_am_t[griddim2, self.blockdim2, self.stream](a, thresh, flip_x, set_val, out)
                elif thresh_dim[0] == a_dim[1]:
                    griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
                    thresh_ma_t[griddim2, self.blockdim2, self.stream](a, thresh, flip_x, set_val, out)
                else:
                    raise ValueError('matrices not aligned')
            elif thresh.shape == ():
                griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
                thresh_m[griddim2, self.blockdim2, self.stream](a, thresh, flip_x, set_val, out)
            else:
                raise ValueError('matrices are not aligned')
        elif a.ndim == 1:
            if a_dim == thresh_dim:
                pass
            elif thresh_dim == ():
                griddim2 = int(ceil(a_dim[0]/self.blockdim))
                thresh_v[griddim2, self.blockdim](a, thresh, flip_x, set_val, out)
            else:
                raise ValueError('matrices are not aligned')
        else:
            raise NotImplementedError

        return a

    def clip(self, a, low, high, out=None):
        """Clip input.

        Parameters
        ----------
        a : array-like
            Array to rectify.
        low : float
            Lowest value for array.
        high : float
            Highest value for array.
        """

        a, out_dtype = _check_array(a)
        a_dim = a.shape

        if type(out) == cuda.cudadrv.devicearray.DeviceNDArray:
            pass
        elif out is None:
            pass
        else:
            raise NotImplementedError
        if out is None:
            out = cuda.device_array(shape=a_dim, dtype=out_dtype, order='F')
        elif out.shape == a_dim:
            pass
        else:
            raise ValueError('matrices are not aligned')

        if a.ndim == 2:
            griddim2 = (int(ceil(a_dim[0]/self.blockdim2[0])),int(ceil(a_dim[1]/self.blockdim2[1])))
            clip_m[griddim2, self.blockdim2, self.stream](a, low, high, out)
        elif a.ndim == 1:
            griddim = int(ceil(a_dim[0]/self.blockdim))
            clip_v[griddim, self.blockdim, self.stream](a, low, high, out)
        else:
            raise NotImplementedError

        return out

    def ones(self, shape, out=None):
        return self.const(shape, 1., out)

    def zeros(self, shape, out=None):
        return self.const(shape, 0., out)

    def const(self, shape, value, out=None):
        if type(shape) != tuple:
            shape = (shape,)
        assert len(shape) > 0
        if len(shape) > 2:
            raise NotImplementedError

        if out is None:
            out = cuda.device_array(shape=shape, dtype=np.float32, order='F')
        if out.shape != shape:
            raise ValueError('matrices are not aligned')

        out_dim = out.shape

        if out.ndim == 2:
            griddim2 = (int(ceil(out_dim[0]/self.blockdim2[0])),int(ceil(out_dim[1]/self.blockdim2[1])))
            const_m[griddim2, self.blockdim2, self.stream](out, value)
        elif out.ndim == 1:
            griddim = int(ceil(out_dim[0]/self.blockdim))
            const_v[griddim, self.blockdim, self.stream](out, value)
        else:
            raise NotImplementedError

        return out

    def reshape(self, a, newshape, order='F'):
        """ Reshapes a to have shape.

        Parameters
        ----------
        a : array-like
            Array to reshape.
        shape : tuple of ints
            Target shape
        """
        a, d_type = _check_array(a)
        assert np.prod(a.shape) == np.prod(newshape)
        return a.reshape(newshape, order=order)


@cuda.jit('void(f4[:,:],f4[:])')
def sum_0(a, out):
    n = a.shape[0]
    m = a.shape[1]
    i = cuda.grid(1)

    if i < m:
        out[i] = 0.
        for jj in xrange(n):
            out[i] = out[i]+a[jj,i]
@cuda.jit('void(f4[:,:],f4[:])')
def sum_1(a, out):
    n = a.shape[0]
    m = a.shape[1]
    i = cuda.grid(1)

    if i < n:
        out[i] = 0.
        for jj in xrange(m):
            out[i] = out[i]+a[i,jj]

@cuda.jit('void(f4[:,:],f4,f4[:])')
def mean_0(a, div, out):
    n = a.shape[0]
    m = a.shape[1]
    i = cuda.grid(1)

    if i < m:
        out[i] = 0.
        for jj in xrange(n):
            out[i] += a[jj,i]
        out[i] = out[i]/div
@cuda.jit('void(f4[:,:],f4,f4[:])')
def mean_1(a, div, out):
    n = a.shape[0]
    m = a.shape[1]
    i = cuda.grid(1)

    if i < n:
        out[i] = 0.
        for jj in xrange(m):
            out[i] += a[i,jj]
        out[i] = out[i]/div

@cuda.jit('void(f4[:,:],f4[:,:])')
def exp_m(a, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = exp(a[i,j])
@cuda.jit('void(f4[:],f4[:])')
def exp_v(a, out):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = exp(a[i])

@cuda.jit('void(f4[:,:],f4[:,:])')
def log_m(a, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = log(a[i,j])
@cuda.jit('void(f4[:],f4[:])')
def log_v(a, out):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = log(a[i])

@cuda.jit('void(f4[:,:],f4[:,:])')
def abs_m(a, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = fabs(a[i,j])
@cuda.jit('void(f4[:],f4[:])')
def abs_v(a, out):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = fabs(a[i])

@cuda.jit('void(f4[:,:],f4[:,:])')
def tanh_m(a, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = tanh(a[i,j])
@cuda.jit('void(f4[:],f4[:])')
def tanh_v(a, out):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = tanh(a[i])

@cuda.jit('void(f4[:,:],f4,f4,f4[:,:])')
def clip_m(a, low, high, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        if a[i,j] < low:
            out[i,j] = low
        elif a[i,j] > high:
            out[i,j] = high
        else:
            out[i,j] = a[i,j]
@cuda.jit('void(f4[:],f4,f4,f4[:])')
def clip_v(a, low, high, out):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        if a[i] < low:
            out[i] = low
        elif a[i] > high:
            out[i] = high
        else:
            out[i] = a[i]

@cuda.jit('void(f4[:,:],f4,int8,f4,f4[:,:])')
def thresh_m(a, thresh, flip_x, set_val, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        if flip_x == 0:
            if a[i,j] < thresh:
                out[i,j] = set_val
        else:
            if a[i,j] > thresh:
                out[i,j] = set_val
@cuda.jit('void(f4[:,:],f4[:,:],int8,f4,f4[:,:])')
def thresh_m_t(a, thresh, flip_x, set_val, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        if flip_x == 0:
            if a[i,j] < thresh[i,j]:
                out[i,j] = set_val
        else:
            if a[i,j] > thresh[i,j]:
                out[i,j] = set_val
@cuda.jit('void(f4[:,:],f4[:,:],int8,f4,f4[:,:])')
def thresh_m_nt(a, thresh, flip_x, set_val, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        if flip_x == 0:
            if a[i,j] < thresh[0,j]:
                out[i,j] = set_val
        else:
            if a[i,j] > thresh[0,j]:
                out[i,j] = set_val
@cuda.jit('void(f4[:,:],f4[:,:],int8,f4,f4[:,:])')
def thresh_m_tn(a, thresh, flip_x, set_val, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        if flip_x == 0:
            if a[i,j] < thresh[i,0]:
                out[i,j] = set_val
        else:
            if a[i,j] > thresh[i,0]:
                out[i,j] = set_val
@cuda.jit('void(f4[:,:],f4[:],int8,f4,f4[:,:])')
def thresh_am_t(a, thresh, flip_x, set_val, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        if flip_x == 0:
            if a[i,j] < thresh[i]:
                out[i,j] = set_val
        else:
            if a[i,j] > thresh[i]:
                out[i,j] = set_val
@cuda.jit('void(f4[:,:],f4[:],int8,f4,f4[:,:])')
def thresh_ma_t(a, thresh, flip_x, set_val, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        if flip_x == 0:
            if a[i,j] < thresh[j]:
                out[i,j] = set_val
        else:
            if a[i,j] > thresh[j]:
                out[i,j] = set_val
@cuda.jit('void(f4[:],f4,int8,f4,f4[:])')
def thresh_v(a, thresh, flip_x, set_val,out):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        if flip_x == 0:
            if a[i] < thresh:
                out[i] = set_val
        else:
            if a[i] > thresh:
                out[i] = set_val
@cuda.jit('void(f4[:],f4[:],int8,f4,f4[:])')
def thresh_v_t(a, thresh, flip_x, set_val,out):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        if flip_x == 0:
            if a[i] < thresh[i]:
                out[i] = set_val
        else:
            if a[i] > thresh[i]:
                out[i] = set_val

@cuda.jit('void(f4[:,:],f4)')
def const_m(out, const):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = const
@cuda.jit('void(f4[:],f4)')
def const_v(out, const):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = const

@cuda.jit('void(f4[:,:])')
def zero_diag_m(out):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i,i] = 0.

@cuda.jit('void(f4[:,:],f4[:])')
def set_diag_s(out, value):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i,i] = value[0]
@cuda.jit('void(f4[:,:],f4[:])')
def set_diag_v(out, value):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i,i] = value[i]

@cuda.jit('void(f4[:],f4[:,:])')
def diag2m(a, out):
    n = out.shape[0]
    m = out.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        if i == j:
            out[i,j] = a[i]
        else:
            out[i,j] = 0.
@cuda.jit('void(f4[:,:],f4[:])')
def diag2v(a, out):
    n = out.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = a[i,i]

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

@cuda.jit('void(f4[:],f4[:],f4[:])')
def vadd_pointwise(a,b,out):
    n = a.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = a[i]+b[i]

@cuda.jit('void(f4[:],f4[:],f4,f4,f4[:])')
def vsadd_pointwise(a,b,alpha,beta,out):
    n = a.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = alpha*a[i]+beta*b[i]

@cuda.jit('void(f4[:,:],f4,f4,f4,f4[:,:])')
def ms_sadd_pointwise(a,b,alpha,beta,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = alpha*a[i,j]+beta*b
@cuda.jit('void(f4[:,:],f4,f4[:,:])')
def ms_add_pointwise(a,b,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = a[i,j]+b

@cuda.jit('void(f4[:],f4,f4,f4,f4[:])')
def vs_sadd_pointwise(a,b,alpha,beta,out):
    n = a.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = alpha*a[i]+beta*b
@cuda.jit('void(f4[:],f4,f4[:])')
def vs_add_pointwise(a,b,out):
    n = a.shape[0]
    i = cuda.grid(1)

    if i < n:
        out[i] = a[i]+b

@cuda.jit('void(f4[:,:],f4[:],f4,f4,f4[:,:])')
def mv0_sadd_pointwise(a,b,alpha,beta,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and n < m:
        out[i,j] = alpha*a[i,j]+beta*b[i]
@cuda.jit('void(f4[:,:],f4[:],f4[:,:])')
def mv0_add_pointwise(a,b,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and n < m:
        out[i,j] = a[i,j]+b[i]

@cuda.jit('void(f4[:,:],f4[:],f4,f4,f4[:,:])')
def mv1_sadd_pointwise(a,b,alpha,beta,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = alpha*a[i,j]+beta*b[j]
@cuda.jit('void(f4[:,:],f4[:],f4[:,:])')
def mv1_add_pointwise(a,b,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = a[i,j]+b[j]

@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:])')
def m_nm_add_pointwise(a,b,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = a[i,j]+b[0,j]
@cuda.jit('void(f4[:,:],f4[:,:],f4,f4,f4[:,:])')
def m_nm_sadd_pointwise(a,b,alpha,beta,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = alpha*a[i,j]+beta*b[0,j]

@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:])')
def m_mn_add_pointwise(a,b,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = a[i,j]+b[i,0]
@cuda.jit('void(f4[:,:],f4[:,:],f4,f4,f4[:,:])')
def m_mn_sadd_pointwise(a,b,alpha,beta,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = alpha*a[i,j]+beta*b[i,0]


def _cu_reshape(d_a, a_shape, a_strides, a_dtype):
    """Reshapes d_a to have same dimensions as a
    
    Parameters
    ----------
    d_a
        Pointer to DeviceNDArray.
    a_shape
        Target shape for array.
    a_strides
        Target strides for array.
    a_dtype
        Target dtype for array.
    """

    if np.prod(d_a.shape) != np.prod(a_shape):
        raise ValueError('total size of new array must be unchanged')
    if d_a.ndim > 2 and len(a_shape) > 2:
        raise NotImplementedError

    out = cuda.devicearray.DeviceNDArray(
            shape=a_shape, strides=a_strides,
            dtype=a_dtype,gpu_data=d_a.gpu_data)
    return out

def _check_array(a):
    """Checks whether array is valid for moving to gpu and moves data to gpu.

    Parameters
    ----------
    a : array-like
        Array to move to gpu
    """
    ok_dtypes = [np.int, np.float32, np.float64]
    if isinstance(a, np.ndarray):
        a = cuda.to_device(np.array(a, dtype=np.float32, order='F'))
    elif isinstance(a, cuda.cudadrv.devicearray.DeviceNDArray):
        pass
    else:
        a = np.array(a)
        if a.dtype not in ok_dtypes:
            raise ValueError('input of type '+str(a.dtype)+
                             ' is not supported')
        else:
            a = np.array(a,dtype=np.float32, order='F')
    if a.dtype == np.float32:
        out_dtype = a.dtype
    else:
        raise NotImplementedError
    return (a, out_dtype)

