""" Numpy-like GPU wrapper. Contains the main gpupy class 
and supporting numbapro cuda kernels.

Jesse Livezey and Zayd Enam
May 7th 2014
"""
from __future__ import division
__authors__   = "Jesse Livezey, Zayd Enam"
__copyright__ = "(c) 2014, Jesse Livezey, Zayd Enam"
__license__   = "The MIT License (MIT)"
__contact__   = "Jesse Livezey <jesse.livezey+gpupy@berkeley.edu>"
import numbapro.cudalib.cublas
from numbapro import cuda
import numpy as np
from math import ceil

class Gpupy(object):
    """Class that has cuBLAS wrappers and additional GPU functionality.
    
    Behaves like numpy functions whenever possible.
    
    Parameters
    ----------
    gpuID : int
        ID of GPU to use. Cannot be changed later and must be set before any
        cuda functionality is used.
    """

    def __init__(self, gpuID=None):
        if gpuID is not None:
            if gpuID < len(cuda.list_devices()) and gpuID >= 0:
                cuda.close()
                cuda.select_device(gpuID)
            else:
                raise ValueError('GPU ID not found')
        self.blas = numbapro.cudalib.cublas.Blas()
        self.stream = cuda.stream()

    def syncronize(self):
        """Synchronize cuda stream."""
        self.stream.synchronize()

    def sync(self):
        """Alias to synchronize."""
        self.synchronize()

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

        b, out_dtype = check_array(b)
        a, out_dtype = check_array(a)

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
        """Returns the transpose of a 2D array.

        Parameters
        ----------
        a : array-like
            Numpy or DeviceNDArray to transpose.
        out : DeviceNDArray (optional)
            Array to overwrite with result.
        """

        a, out_dtype = check_array(a)
            
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

        b, out_dtype = check_array(b)
        a, out_dtype = check_array(a)

        if out == cuda.cudadrv.devicearray.DeviceNDArray:
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
            # np.newaxis matricies
            elif a_dim[0] == b_dim[0] and b_dim[1] == 0:
                if out is None:
                    out = cuda.device_array((a_dim[0], a_dim[1]), dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0] and out.shape[1] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                blockdim = 32
                griddim = int(ceil(a_dim[0]/blockdim))
                if alpha != 1. or beta != 1.:
                    mv0f_sadd_pointwise(a,b,alpha,beta,out)
                else:
                    mv0f_add_pointwise(a,b,out)
            elif a_dim[1] == b_dim[1] and b_dim[0] == 0:
                if out is None:
                    out = cuda.device_array((a_dim[0], a_dim[1]), dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0] and out.shape[1] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                blockdim = 32
                griddim = int(ceil(a_dim[1]/blockdim))
                if alpha != 1. or beta != 1.:
                    mv1f_sadd_pointwise(a,b,alpha,beta,out)
                else:
                    mv1f_add_pointwise(a,b,out)
            elif b_dim[0] == a_dim[0] and a_dim[1] == 0:
                if out is None:
                    out = cuda.device_array((b_dim[0], b_dim[1]), dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0] and out.shape[1] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                blockdim = 32
                griddim = int(ceil(b_dim[0]/blockdim))
                if alpha != 1. or beta != 1.:
                    mv0f_sadd_pointwise(b,a,beta,alpha,out)
                else:
                    mv0f_add_pointwise(b,a,out)
            elif b_dim[1] == a_dim[1] and a_dim[0] == 0:
                if out is None:
                    out = cuda.device_array((b_dim[0], b_dim[1]), dtype=out_dtype, order='F')
                elif out.shape[0] == a_dim[0] and out.shape[1] == a_dim[1]:
                    pass
                else:
                    raise ValueError('matrices are not aligned')
                blockdim = 32
                griddim = int(ceil(b_dim[1]/blockdim))
                if alpha != 1. or beta != 1.:
                    mv1f_sadd_pointwise(b,a,beta,alpha,out)
                else:
                    mv1f_add_pointwise(b,a,out)
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
                vsadd_pointwise[griddim,blockdim](a,b,alpha,beta,out)
            else:
                vadd_pointwise[griddim,blockdim](a,b,out)
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
                ms_sadd_pointwise[griddim,blockdim](a,b,alpha,beta,out)
            else:
                ms_add_pointwise[griddim,blockdim](a,b,out)
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
                ms_sadd_pointwise[griddim,blockdim](b,a,beta,alpha,out)
            else:
                ms_add_pointwise[griddim,blockdim](b,a,out)
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
                vs_sadd_pointwise[griddim,blockdim](a,b,alpha,beta,out)
            else:
                vs_add_pointwise[griddim,blockdim](a,b,out)
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
                vs_sadd_pointwise[griddim,blockdim](b,a,beta,alpha,out)
            else:
                vs_add_pointwise[griddim,blockdim](b,a,out)
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
                    mv0_sadd_pointwise[griddim,blockdim](a,b,alpha,beta,out)
                else:
                    mv0_add_pointwise[griddim,blockdim](a,b,out)
            elif b.shape[0] == a.shape[1]:
                if alpha != 1. or beta != 1.:
                    mv1_sadd_pointwise[griddim,blockdim](a,b,alpha,beta,out)
                else:
                    mv1_add_pointwise[griddim,blockdim](a,b,out)
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
                    mv0_sadd_pointwise[griddim,blockdim](b,a,beta,alpha,out)
                else:
                    mv0_add_pointwise[griddim,blockdim](b,a,out)
            elif a.shape[0] == b.shape[1]:
                if alpha != 1. or beta != 1.:
                    mv1_sadd_pointwise[griddim,blockdim](b,a,beta,alpha,out)
                else:
                    mv1_add_pointwise[griddim,blockdim](b,a,out)
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

        b, out_dtype = check_array(b)
        a, out_dtype = check_array(a)

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
            griddim2 = (int(ceil(a_dim[0]/blockdim2[0])),int(ceil(a_dim[1]/blockdim2[1])))
            print 
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
            griddim = int(ceil(a_dim[0]/blockdim))
            print griddim
            vmultiply_pointwise[griddim,blockdim](a,b,out)
        else:
            raise NotImplementedError

        return out

    def sub(self, a, b, out = None, alpha = 1., beta = 1.):
        return  self.add(a, b, alpha=alpha, beta=-beta)

    def scal(self, a, alpha):
        """Scale a 1D or 2D array by alpha.

        Parameters
        ----------
        a : array-like
            Array to scale.
        alpha : float
            Scaling factor.
        """

        a, out_dtype = check_array(a)

        a_dim = a.shape

        if a.ndim == 2:
            a_strides = a.strides
            a_dtype = a.dtype
            d_flat_a = cu_reshape(a, (np.prod(a_dim),), (a_strides[0],), a_dtype)
            self.blas.scal(alpha, d_flat_a)
            a = cu_reshape(d_flat_a, a_dim, a_strides, a_dtype)
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
        raise NotImplementedError

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
        raise NotImplementedError

    def diag(self, a, out=None):
        raise NotImplementedError

    def zero_diag(self, a, out=None):
        raise NotImplementedError

    def zero_diag(self, a, out=None):
        raise NotImplementedError

    def relu(self, a, thresh=0., neg=False, out=None):
        raise NotImplementedError

    def ones(shape, out=None):
        raise NotImplementedError

    def zeros(shape, out=None):
        raise NotImplementedError

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

@cuda.jit('void(f4[:,:],f4[:,:],f4,f4,f4[:,:])')
def mv0f_sadd_pointwise(a,b,alpha,beta,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and n < m:
        out[i,j] = alpha*a[i,j]+beta*b[i,0]
@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:])')
def mv0f_add_pointwise(a,b,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and n < m:
        out[i,j] = a[i,j]+b[i,0]

@cuda.jit('void(f4[:,:],f4[:,:],f4,f4,f4[:,:])')
def mv1f_sadd_pointwise(a,b,alpha,beta,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = alpha*a[i,j]+beta*b[0,j]
@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:])')
def mv1f_add_pointwise(a,b,out):
    n = a.shape[0]
    m = a.shape[1]
    i,j = cuda.grid(2)

    if i < n and j < m:
        out[i,j] = a[i,j]+b[0,j]

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

def cu_reshape(d_a, a_shape, a_strides, a_dtype):
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

def check_array(a):
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

