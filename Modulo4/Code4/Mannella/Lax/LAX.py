import numpy as np
from numba import jit_module

def LAX_complete(evo, u, Nt, alpha, ninner):
    evo[0] = u
    for i in range(Nt-1):
        u = LAX(u, alpha, ninner)
        evo[i+1] = u
    return evo

def LAX(u, alpha, ninner):
    for i in range(ninner):
        u[1:-1] = LAX_step(u[:-2], u[2:], alpha)
        u[0] = LAX_step(u[-1], u[1], alpha)
        u[-1] = LAX_step(u[-2], u[0], alpha)
    return u

def LAX_step(u_prev, u_next, alpha):
    return 1/2 * (u_next * ( 1 - alpha ) + u_prev * ( 1 + alpha ))

jit_module(nopython = True, cache = True, fastmath = True)
