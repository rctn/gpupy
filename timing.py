from __future__ import division
import numpy as np
from timeit import default_timer as timer
from numbapro import cuda
from gpupy import Gpupy


""" Times various gpupy functions.
"""

nshort = 6
tshort = 2
nmed = 3
tmed = 6
nlong = 1
  
#Setup variables for testing
dim = 4096
dimMatrix = (dim,dim)
d_type = np.float32
nIter = 10
    
params = """Parameters for dot:
         Matrix Size: """+str(dimMatrix)+"""
         nIter: """+str(nIter)+"""\n"""
print params
             
rng = np.random.RandomState(0)
start = timer()
matrix1 = np.array(rng.rand(*dimMatrix),dtype=d_type, order='F')
matrix2 = np.array(rng.rand(*dimMatrix),dtype=d_type, order='F')
matrix3_np = np.zeros(shape=dimMatrix,dtype=d_type, order='F')
dt = timer()-start
print '---------------Numpy based dot---------------'
print 'Time to create arrays:'
print '%f s' % dt
start = timer()
for ii in xrange(nIter):
    matrix3_np[:] = np.dot(matrix1, matrix2)
dt = timer()-start
mult = dt
print 'Time for matrix dot:'
print '%f s' % (dt/float(nIter))
print 'Teraflops:'
print 2.*dim**3/float(dt/float(nIter))/1.e12

gp = Gpupy()
stream = gp.stream
rng = np.random.RandomState(0)
start = timer()
matrix1 = cuda.to_device(np.array(rng.rand(*dimMatrix),dtype=d_type, order='F'), stream=stream)
matrix2 = cuda.to_device(np.array(rng.rand(*dimMatrix),dtype=d_type, order='F'), stream=stream)
matrix3_gp = cuda.to_device(np.zeros(shape=dimMatrix,dtype=d_type,order='F'), stream=stream)
stream.synchronize()
dt = timer()-start
print '-----------NumbaPro GPU based dot------------'
print 'Time to create arrays:'
print '%f s' % dt
start = timer()
for ii in xrange(nIter):
    gp.dot(matrix1, matrix2, out=matrix3_gp)
gp.sync()
dt = timer()-start
mult = mult/dt
print 'Time for matrix dot:'
print '%f s' % (dt/float(nIter))
print 'Teraflops:'
print 2.*dim**3/float(dt/float(nIter))/1.e12
start = timer()
matrix3_gp = matrix3_gp.copy_to_host()
dt = timer()-start
print 'Time to transer results to host:'
print '%f s' % dt
assert np.allclose(matrix3_gp, matrix3_np), "dot products not returning same answer"
print str(mult)+' times speedup'
    

nIter = 100
params = """Parameters for add:
         Matrix Size: """+str(dimMatrix)+"""
         nIter: """+str(nIter)+"""\n"""
print ''
print '---------------------------------------------'
print ''
print params
             
rng = np.random.RandomState(0)
start = timer()
matrix1 = np.array(rng.rand(*dimMatrix), dtype=d_type, order='F')
matrix2 = np.array(rng.rand(*dimMatrix), dtype=d_type, order='F')
matrix3_np = np.zeros(shape=dimMatrix, dtype=d_type, order='F')
dt = timer()-start
print '---------------Numpy based add---------------'
print 'Time to create arrays:'
print '%f s' % dt
start = timer()
for ii in xrange(nIter):
    matrix3_np[:] = np.add(matrix1, matrix2)
dt = timer()-start
mult = dt
print 'Time for matrix add:'
print '%f s' % (dt/float(nIter))
print 'Teraflops:'
print dim**2/float(dt/float(nIter))/1.e12

gp = Gpupy()
stream = gp.stream
rng = np.random.RandomState(0)
start = timer()
matrix1 = cuda.to_device(np.array(rng.rand(*dimMatrix), dtype=d_type, order='F'), stream=stream)
matrix2 = cuda.to_device(np.array(rng.rand(*dimMatrix), dtype=d_type, order='F'), stream=stream)
matrix3_gp = cuda.to_device(np.zeros(shape=dimMatrix, dtype=d_type,order='F'), stream=stream)
stream.synchronize()
dt = timer()-start
print '-----------NumbaPro GPU based add------------'
print 'Time to create arrays:'
print '%f s' % dt
start = timer()
for ii in xrange(nIter):
    gp.add(matrix1, matrix2, out=matrix3_gp)
gp.sync()
dt = timer()-start
mult = mult/dt
print 'Time for matrix add:'
print '%f s' % (dt/float(nIter))
print 'Teraflops:'
print dim**2/float(dt/float(nIter))/1.e12
start = timer()
matrix3_gp = matrix3_gp.copy_to_host()
dt = timer()-start
print 'Time to transer results to host:'
print '%f s' % dt
assert np.allclose(matrix3_gp, matrix3_np), "add not returning same results"
print str(mult)+' times speedup'
