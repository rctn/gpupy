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

### Speedup on Tesla K40 vs. Xeon E-5 1620 (with MKL)
20x for dot

30x for add

```
Parameters for dot:
         Matrix Size: (4096, 4096)
         nIter: 10

---------------Numpy based dot---------------
Time to create arrays:
0.812318 s
Time for 10 dots:
8.985346 s
-----------NumbaPro GPU based dot------------
Time to create arrays:
0.865464 s
Time for 10 dots:
0.457248 s
Time to transer results to host:
0.028134 s
19.6509260172 times speedup

---------------------------------------------

Parameters for add:
         Matrix Size: (4096, 4096)
         nIter: 100

---------------Numpy based add---------------
Time to create arrays:
0.824676 s
Time for 100 adds:
3.414449 s
-----------NumbaPro GPU based add------------
Time to create arrays:
0.856417 s
Time for 100 adds:
0.114286 s
Time to transer results to host:
0.028349 s
29.8763679983 times speedup
```
