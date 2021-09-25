from numba import jit, prange
import numpy as np

#def integrate(f, step, h, p = (), df=None, kind='trapezoidal'):
#    x = np.empty( ( len(x0) , step ) ).ravel()
#    for i in range( len(x0) ):
#        for j in range(step-1):
#            x[i,j+1] += x[i,j] + h*f(x[i,j], p)
#    return x
#

def slow_easy_integrate(f, x0, step, h, kind='trapezoidal'):
    x = np.empty(step)
    x[0] = x0
    for i in range(step-1):
        x[i+1] += x[i] + h * f(x[i])
    return x


@jit(parallel = True, fastmath = True)
def fast_easy_integrate(f, x0, step, h, kind='trapezoidal'):
    x = np.empty(step)
    x[0] = x0
    for i in range(step-1):
        x[i+1] += x[i] + h * f(x[i])
    return x


#def stepper(f, x, step, h):
#    for i in prange( len(x0) ):
#        for j in range(step-1):
#            x[i,j+1] += x[i,j] + h*f(x[i,j], p)
#    return x

