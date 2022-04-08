import time
import numpy as np
from random import random
cimport numpy as np
cimport cython
from libc.math cimport exp

ctypedef np.double_t DTYPE_t

def do_calc(int nstat, float start, float aver, 
            float sigma, float delta): #numpy.ndarray[DTYPE_t, ndim=1] arr
    """
    Metropolis Algorithm for sample a gaussian distribution.

    Parameters
    ----------
    nstat: int
        Number of measures of the output.
    start: float
        Starting value of x.
    aver: float
        Average of the gaussian distribution.
    sigma: float
        Sigma of the gaussian distribution.
    delta:
        Amplitude max of the step for the metropolis.

    Returns
    -------
    (1d numpy array[float], 1d numpy array[int])
        (sampling of the simulation, acceptance of the sample)
    """
    rng =  np.random.Generator(np.random.PCG64())

    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] val_arr = np.empty(nstat)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rand_arr = rng.uniform(size = (nstat, 2))
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] acc_arr = np.empty(nstat).astype(int)

    metro_loop(nstat, start, delta, aver, sigma, val_arr, rand_arr, acc_arr)
    np.savetxt(f'data/data.dat', np.column_stack((val_arr, acc_arr))) # Save Energy and Magnetization
    return val_arr, acc_arr

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)  
cdef void metro_loop(int nstat, float start, float delta, float aver, float sigma, 
                     DTYPE_t[:] val_arr, DTYPE_t[:, :] rand_arr, np.int_t[:] acc_arr):
    cdef int i, acc
    cdef float z, q_try, q = start
    cdef float sigma2 = sigma * sigma

    for i in range(nstat):

        q_try = q + delta * ( 1. - 2 * rand_arr[i, 0] )
        z = exp((( q - aver )**2 - ( q_try - aver )**2)/(2. * sigma2))
        
        if rand_arr[i, 1] < z:
            q = q_try
            acc = 1
        else: acc = 0
        acc_arr[i] = acc
        val_arr[i] = q
