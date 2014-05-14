gpupy
=====

Wrapper for GPU cuBLAS\NumbaPro functions that provides drop in support for NumPy functions

Requires:
- NumPy
- NumbaPro
- CUDA installation
- CUDA capable GPU
- nose

### Use
Instantiate a Gpupy object 
```python 
from gpupy import Gpupy
gp = Gpupy()
out = gp.function(*args)
```
#### Functions:

- gp.dot: np.dot drop-in (matrix-matrix and matrix-vector products)
- gp.T: np.transpose drop-in (matrix)
- gp.add: np.add drop-in (matrix-matrix and vector-vector addition)
- gp.mult: np.multiply drop-in (matrix-matrix and vector-vector multiplication)

### Speedup on Tesla K40 vs. Xeon E-5 1620

```
Parameters:
             Matrix Size: (8192, 8192)
             nIter: 10

---------------Numpy based dot---------------
Time to create arrays:
1.973649 s
Time for 10 dots:
69.731279 s
-----------NumbaPro GPU based dot------------
Time to create arrays:
3.788626 s
Average time over 10 dots:
3.224349 s
```
