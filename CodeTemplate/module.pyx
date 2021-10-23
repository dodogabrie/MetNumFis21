import numpy as np
cimport cython, numpy as np # Make numpy work with cython
from libc.math cimport sin  # How to import C function
#from libc.stdio cimport printf
#printf("%d\r", i)

ctypedef np.double_t DTYPE_t

def fname(int N):
    """
    Description
    
    Parameters
    ----------

    Returns
    -------
    """
    # Define a variable
    cdef float pi = np.pi
    # Define an array
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] f = np.empty(N)
    #if write: 
    #    np.savetxt(f'data/data.dat', np.column_stack((x, f))) # x and sin(x)
 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Make division fast like C
cdef void for_fill(DTYPE_t[:] f):
    return 
