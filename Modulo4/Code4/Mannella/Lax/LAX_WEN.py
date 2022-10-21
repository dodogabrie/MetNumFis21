import numpy as np
from numba import jit_module

def LAX_WEN_complete(evo, u, Nt, alpha, ninner):
    evo[0] = u
    for i in range(Nt-1):
        u = LAX_WEN(u, alpha, ninner)
        evo[i+1] = u
    return evo

def LAX_WEN(u, alpha, ninner, Nt = 2):
    for j in range(Nt-1):
       for i in range(ninner):
           w = np.copy(u)
           u[:-1] = LAX_step(u[:-1], u[1:], alpha)
           u[1:-1] = w[1:-1] - alpha * (u[1:-1] - u[:-2])
           u[0] = u[-2]
           u[-1] = u[1]
    return u

def LAX_step(u_prev, u_next, alpha):
    return 1/2 * (u_next * ( 1 - alpha ) + u_prev * ( 1 + alpha ))

def periodic_LF_boundary(u, alpha):
    xi = u[0] - alpha * (u[0] - u[-1])
    xf = u[-1] - alpha * (u[-1] - u[-2])
    return xi, xf



jit_module(nopython = True, cache = True, fastmath = True)
