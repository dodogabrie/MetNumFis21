from numba import njit, prange
import numpy as np

@njit(fastmath = True)
def euler(f, x, t, p = ()):
    h = t[1]-t[0]
    step = len(t)
    for j in range(step-1):
        x[j+1] = x[j] + h * f(x[j], t[j], p)
    return x

@njit(fastmath = True)
def trapezoid(f, x, t, p = ()):
    h = t[1]-t[0]
    step = len(t)
    for j in range(step-1):
        f_here = f( x[j], t[j] , p )
        x_next = x[j] + h * f_here
        f_next_euler = f( x_next, t[j+1] , p)
        x[j+1] = x[j] + h/2 * ( f_here + f_next_euler )
        f_next = f( x[j+1], t[j+1] , p )
        x[j+1] = x[j] + h/2 * ( f_here + f_next )
    return x

@njit(fastmath = True)
def AB(f, x, t, p = ()):
    h = t[1]-t[0]
    step = len(t)
    f_here = f(x[0], t[0], p)
    x[1] = x[0] + h * f_here
    f_back = f_here
    for j in range(step-1):
        f_here = f(x[j], t[j], p)
        x[j+1] = x[j] + h/2 * (3 * f_here - f_back)
        f_back = f_here
    return x

@njit(fastmath = True)
def midpoint(f, x, t, p = ()):
    h = t[1]-t[0]
    step = len(t)
    for j in range(step-1):
        f_here = f(x[j], t[j], p = ())
        if j == 0:
            x[j+1] = x[j] + h * f_here
        else:
            x[j+1] = x[j-1] + 2 * h * f_here 
    return x


@njit(fastmath = True)
def RK45(f, x, t, p = ()):
    h = t[1]-t[0]
    step = len(t)
    for j in range(step-1):
        k1 = h * f( x[j] , t[j] , p )
        k2 = h * f( x[j] + k1 / 2 , t[j] + h / 2 , p)
        k3 = h * f( x[j] + k2 / 2 , t[j] + h / 2 , p)
        k4 = h * f( t[j] + h , x[j] + k3 , p)
        x[j+1] = x[j] + k1/6 + k2/3 + k3 / 3 + k4 / 6
    return x
