gpupy
=====

Wrapper for GPU cuBlas\NumbaPro functions that provides drop in support for NumPy functions

### Use
Instantiate a Gpupy object 
```python 
from gpupy import Gpupy
gp = Gpupy()
```
Functions:

gp.dot - np.dot drop-in (matrix-matrix and matrix-vector products)
