"""
This module implement the sin(x) function using Cython and Numpy.
"""

import numpy as np
import math
cimport cython, numpy as np # Make numpy work with cython
from cython.parallel import prange
from libc.math cimport sin
#from libc.stdio cimport printf

ctypedef np.double_t DTYPE_t

def MySin(int Lx, int N, int write = 0):
    """
    Return vector containing the function f(x) = sin(x) given the spatial
    domain (periodic) in term of 2*pi. 
    
    Parameters
    ----------
    Lx : int
        Number of period of the x domain.
    N : int
        Number of points in which we evaluate f(x) inside the domain 
        defined by Lx. This domain is equally divided in piece of 
        dx = Lx/N * 2*pi
    write: int
        If 1 write on file 'data/data.dat', otherwise just return array 
        of results.

    Returns
    -------
    (1d numpy array, 1d numpy array)
    """
    cdef float pi = np.pi
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] x = np.linspace(0, Lx * 2 * pi, N)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] f = np.empty(N)
    fill_f(x, f, N)
    if write: 
        np.savetxt(f'data/data.dat', np.column_stack((x, f))) # x and sin(x)
    return x, f
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void fill_f(DTYPE_t[:] x, DTYPE_t[:] f, int N):
    cdef int i
    for i in prange(N, nogil=True):
        f[i] = sin(x[i])
