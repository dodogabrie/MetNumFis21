import numpy as np
cimport cython, numpy as np # Make numpy work with cython

ctypedef np.double_t DTYPE_t

def solve_tridiag(DTYPE_t[:] diag, DTYPE_t[:] dlo, DTYPE_t[:] dup, DTYPE_t[:] b):
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
        Array of b of the system.

    Returns
    -------
    (1d numpy array, boolean)
        (the solution, invertibility of A)
    """
    cdef int inv, N = len(b)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] x = np.zeros(N).astype(float)
    inv = gauss_red(diag, dlo, dup, b, N)
    if inv != 0:
        inv = solve(diag, dup, b, x, N)
    return x, bool(inv)

 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Make division fast like C
cdef int gauss_red(DTYPE_t[:] diag, DTYPE_t[:] dlo, DTYPE_t[:] dup, DTYPE_t[:] b, int N):
    cdef int i, inv = 1
    cdef float factor
    for i in range(N-1):
        if diag[i] == 0:
            inv = 0
            break
        else:
            factor = dlo[i]/diag[i]
            diag[i+1] = diag[i+1] - factor * dup[i]
            b[i+1] = b[i+1] - factor * b[i]
    return inv

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Make division fast like C
cdef int solve(DTYPE_t[:] diag, DTYPE_t[:] dup, DTYPE_t[:] b, DTYPE_t[:] x, int N):
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
