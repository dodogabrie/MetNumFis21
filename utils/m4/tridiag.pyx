""" 
This module solve a Linear System of equation that can be written in the form 
of Ax = b with A tridiagonal.
"""

import numpy as np
cimport cython, numpy as np # Make numpy work with cython

ctypedef np.double_t DTYPE_t

def solve(DTYPE_t[:] diag, DTYPE_t[:] dlo, DTYPE_t[:] dup, DTYPE_t[:] b):
    """
    Solve linear system Ax = b with A tridiagonal.
    
    Parameters
    ----------
    diag : 1d numpy array
        Diagonal elements of A
    dlo : 1d numpy array
        Subdiagonal element of A (len(diag)-1)
    dup : 1d numpy array
        Supdiagonal element of A (len(diag)-1)
    b : 1d numpy array
        Array b of the system.

    Returns
    -------
    (1d numpy array, boolean)
        (the solution, invertibility of A)
    """
    cdef int inv, N = len(b)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] x = np.empty(N).astype(float)
    inv = gauss_reduction(diag, dlo, dup, b, N)
    if inv != 0:
        inv = find_solution(diag, dup, b, x, N)
    return x, bool(inv)

 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Make division fast like C
cdef int gauss_reduction(DTYPE_t[:] diag, DTYPE_t[:] dlo, DTYPE_t[:] dup, DTYPE_t[:] b, int N):
    cdef int i, inv = 1
    cdef DTYPE_t factor
    for i in range(1, N):
        if diag[i-1] == 0:
            inv = 0
            break
        else:
            factor = dlo[i-1]/diag[i-1]
            diag[i] = diag[i] - factor * dup[i-1]
            b[i] = b[i] - factor * b[i-1]
    return inv

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Make division fast like C
cdef int find_solution(DTYPE_t[:] diag, DTYPE_t[:] dup, DTYPE_t[:] b, DTYPE_t[:] x, int N):
    cdef int i, inv = 1
    if diag[N-1] == 0: 
        inv = 0
    else:
        x[N-1] = b[N-1]/diag[N-1]
        for i in range(N-2, -1, -1):
            if diag[i] == 0: 
                inv = 0
                break
            b[i] = b[i] - dup[i] * x[i+1]
            x[i] = b[i]/diag[i]
    return inv
