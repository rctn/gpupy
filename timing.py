#This file will time various versions of LCA
from __future__ import division
import numpy as np
from timeit import default_timer as timer
from numbapro import cuda
from gpupy import Gpupy


"""Times various versions of Matching Pursuit."""

nshort = 6
tshort = 2
nmed = 3
tmed = 6
nlong = 1
  
#Setup variables for testing
dimMatrix = (4096,4096)
d_type = np.float32
nIter = 10
    
params = """Parameters for dot:
         Matrix Size: """+str(dimMatrix)+"""
         nIter: """+str(nIter)+"""\n"""
print params
             
start = timer()
matrix1 = np.array(np.random.rand(*dimMatrix),dtype=d_type)
matrix2 = np.array(np.random.rand(*dimMatrix),dtype=d_type)
dt = timer()-start
print '---------------Numpy based dot---------------'
print 'Time to create arrays:'
print '%f s' % dt
start = timer()
for ii in xrange(nIter):
    np.dot(matrix1,matrix2)
dt = timer()-start
print 'Time for '+str(nIter)+' dots:'
print '%f s' % dt

gp = Gpupy()
start = timer()
matrix1 = cuda.to_device(np.array(np.random.rand(*dimMatrix),dtype=d_type,order='F'))
matrix2 = cuda.to_device(np.array(np.random.rand(*dimMatrix),dtype=d_type,order='F'))
dt = timer()-start
print '-----------NumbaPro GPU based dot------------'
print 'Time to create arrays:'
print '%f s' % dt
start = timer()
for ii in xrange(nIter):
    gp.dot(matrix1,matrix2)
dt = timer()-start
print 'Time for '+str(nIter)+' dots:'
print '%f s' % dt
    

nIter = 100
params = """Parameters for add:
         Matrix Size: """+str(dimMatrix)+"""
         nIter: """+str(nIter)+"""\n"""
print ''
print '---------------------------------------------'
print ''
print params
             
start = timer()
matrix1 = np.array(np.random.rand(*dimMatrix),dtype=d_type)
matrix2 = np.array(np.random.rand(*dimMatrix),dtype=d_type)
dt = timer()-start
print '---------------Numpy based add---------------'
print 'Time to create arrays:'
print '%f s' % dt
start = timer()
for ii in xrange(nIter):
    np.add(matrix1,matrix2)
dt = timer()-start
print 'Time for '+str(nIter)+' adds:'
print '%f s' % dt

gp = Gpupy()
start = timer()
matrix1 = cuda.to_device(np.array(np.random.rand(*dimMatrix),dtype=d_type,order='F'))
matrix2 = cuda.to_device(np.array(np.random.rand(*dimMatrix),dtype=d_type,order='F'))
dt = timer()-start
print '-----------NumbaPro GPU based add------------'
print 'Time to create arrays:'
print '%f s' % dt
start = timer()
for ii in xrange(nIter):
    gp.add(matrix1,matrix2)
dt = timer()-start
print 'Time for '+str(nIter)+' adds:'
print '%f s' % dt
