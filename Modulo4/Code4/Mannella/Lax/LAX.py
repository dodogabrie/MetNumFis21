import numpy as np
from numba import jit_module

def LAX(u, alpha, ninner):
    for i in range(ninner):
        u[1:-1] = 1/2 * (u[2:] * ( 1 - alpha ) + u[:-2] * ( 1 + alpha ))
        u[0] = 1/2 * (u[1] * ( 1 - alpha ) + u[-2] * ( 1 + alpha ))
        u[-1] = 1/2 * (u[0] * ( 1 - alpha ) + u[-2] * ( 1 + alpha ))
    return u

jit_module(nopython = True, cache = True, fastmath = True)
