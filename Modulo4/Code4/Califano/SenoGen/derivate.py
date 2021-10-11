"""
Evaluate the derivative of a function in 3 different way using numpy vectorial
operations. 
The efficient one ( o[(dx)^2] ) is simm_der (the last one), the other 2 methods
are just o[dx].
"""

import numpy as np
#from numba import jit_module

def foward_der(x):
    der = np.empty(len(x))
    dx = x[1]-x[0]
    der[:-1] = (x[1:] - x[:-1])/dx
    der[-1] = der[-2]
    return der

def backward_der(x):
    der = np.empty(len(x))
    dx = x[1]-x[0]
    der[1:] = (x[1:] - x[:-1])/dx
    der[0] = der[1]
    return der

def simm_der(x):
    der = np.empty(len(x))
    dx = x[1]-x[0]
    der[1:-1] = (x[2:] - x[:-2])/(2*dx)
    der[0] = der[1]
    der[-1] = der[-2]
    return der
