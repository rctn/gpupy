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
Functions:

- gp.dot: np.dot drop-in (matrix-matrix and matrix-vector products)
- gp.T: np.transpose drop-in (matrix)
- gp.add: np.add drop-in (matrix-matrix and vector-vector addition)
- gp.mult: np.multiply drop-in (matrix-matrix and vector-vector multiplication)
