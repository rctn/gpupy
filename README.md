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

15x for add

```
Parameters for dot:
         Matrix Size: (4096, 4096)
         nIter: 10

---------------Numpy based dot---------------
Time to create arrays:
0.482110 s
Time for 10 dots:
7.651291 s
-----------NumbaPro GPU based dot------------
Time to create arrays:
0.857696 s
Time for 10 dots:
0.412212 s

---------------------------------------------

Parameters for add:
         Matrix Size: (4096, 4096)
         nIter: 100

---------------Numpy based add---------------
Time to create arrays:
0.484548 s
Time for 100 dots:
2.456518 s
-----------NumbaPro GPU based add------------
Time to create arrays:
0.837742 s
Time for 100 dots:
0.163046 s
```
