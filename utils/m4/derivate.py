"""
Evaluate the derivative of a function in 3 different way using numpy vectorial
operations.
The efficient one ( o[(dx)^2] ) is simm_der (the last one), the other 2 methods
are just o[dx].
"""

import numpy as np
#from numba import jit_module

def foward_der(u,dx):
    """
    Derivative considering the next point
    """
    der = np.empty(len(u))
    der[:-1] = (u[1:] - u[:-1])/dx
    der[-1] = der[0]
    return der

def backward_der(u,dx):
    """
    Derivative considering the previous point
    """
    der = np.empty(len(u))
    der[1:] = (u[1:] - u[:-1])/dx
    der[0] = der[-1]
    return der

def simm_der(f, dx, out):
    """
    Derivative considering both the next and the previous points
    """
    bound0, bound1 = f[0], f[-1]
    bound_next0, bound_next1 = f[1], f[-2]
    out[1:-1] = (f[2:] - f[:-2])/(2*dx)
    out[0] = (bound_next0 - bound1)/(2*dx)
    out[-1] = (bound0 - bound_next1)/(2*dx)
    return out

def simm_der2(f, dx, out):
    """
    Simmetrical second derivative
    """
    b0, b1 = f[0], f[-1]
    bn0, bn1 = f[1], f[-2]
    out[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2])/(dx**2)
    out[0] = (bn0 - 2 * f[0] +  b1)/(dx**2)
    out[-1] = (b0 - 2 * f[-1] + bn1)/(dx**2)
    return out

def shift_test(f, shift_f):
    """
    Testing the continuity of a function under shift
    """
    N = len(f)
    quarter = int(len(f)/4)
    remain = N-quarter
    shift_f[:quarter] = f[-quarter:]
    shift_f[-remain:] = f[:remain]
    return shift_f
