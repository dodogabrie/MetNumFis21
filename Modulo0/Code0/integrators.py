"""Set of functions for the integration of dynamical systems described by ODE (compiled with numba)"""

from numba import njit
import numpy as np

@njit(fastmath = True)
def euler(f, x, t, p = ()):
    """
    Simple Euler integrator.

    Parameters
    ----------
    f: function
        the function to integrate: f(x, t, p=()). 
        The first argument is the real value(s) at this step, 
        the second is the time at this step, the third are a
        tuple of parameters of the system.
    x: numpy nd-array
        Array containing in the first position the initial 
        condition.
    t: numpy nd-array
        Array of the same lenght of x containing the time steps
        in wich the system will be evaluated.
    p: tuple
        Tuple of parameters of the function f.

    Result
    ------
    numpy nd-array
        Array x containing the results of the integration at 
        each time steps of t.
    """
    h = t[1]-t[0]
    step = len(t)
    for j in range(step-1):
        xj = x[j]
        x[j+1] = xj + h * f(xj, t[j], p)
    return x

@njit(fastmath = True)
def trapezoid(f, x, t, p = ()):
    """
    Trapezoid integration method.

    Parameters
    ----------
    f: function
        the function to integrate: f(x, t, p=()). 
        The first argument is the real value(s) at this step, 
        the second is the time at this step, the third are a
        tuple of parameters of the system.
    x: numpy nd-array
        Array containing in the first position the initial 
        condition.
    t: numpy nd-array
        Array of the same lenght of x containing the time steps
        in wich the system will be evaluated.
    p: tuple
        Tuple of parameters of the function f.

    Result
    ------
    numpy nd-array
        Array x containing the results of the integration at 
        each time steps of t.
    """
    h = t[1]-t[0]
    step = len(t)
    for j in range(step-1):
        xj = x[j]
        tj = t[j]
        tjn = t[j+1]
        f_here = f( xj, tj , p )
        x_next = xj + h * f_here
        f_next_euler = f( x_next, tjn , p)
        x[j+1] = xj + h/2 * ( f_here + f_next_euler )
        f_next = f( x[j+1], tjn , p )
        x[j+1] = xj + h/2 * ( f_here + f_next )
    return x

@njit(fastmath = True)
def AB(f, x, t, p = ()):
    """
    AB integration method.

    Parameters
    ----------
    f: function
        the function to integrate: f(x, t, p=()). 
        The first argument is the real value(s) at this step, 
        the second is the time at this step, the third are a
        tuple of parameters of the system.
    x: numpy nd-array
        Array containing in the first position the initial 
        condition.
    t: numpy nd-array
        Array of the same lenght of x containing the time steps
        in wich the system will be evaluated.
    p: tuple
        Tuple of parameters of the function f.

    Result
    ------
    numpy nd-array
        Array x containing the results of the integration at 
        each time steps of t.
    """
    h = t[1]-t[0]
    step = len(t)
    f_here = f(x[0], t[0], p)
    x[1] = x[0] + h * f_here
    f_back = f_here
    for j in range(step-1):
        xj = x[j]
        f_here = f(xj, t[j], p)
        x[j+1] = xj + h/2 * (3 * f_here - f_back)
        f_back = f_here
    return x

@njit(fastmath = True)
def midpoint(f, x, t, p = ()):
    """
    Midpoint integration method.

    Parameters
    ----------
    f: function
        the function to integrate: f(x, t, p=()). 
        The first argument is the real value(s) at this step, 
        the second is the time at this step, the third are a
        tuple of parameters of the system.
    x: numpy nd-array
        Array containing in the first position the initial 
        condition.
    t: numpy nd-array
        Array of the same lenght of x containing the time steps
        in wich the system will be evaluated.
    p: tuple
        Tuple of parameters of the function f.

    Result
    ------
    numpy nd-array
        Array x containing the results of the integration at 
        each time steps of t.
    """
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
    """
    Runge Kutta (of order 4-5) integration method.

    Parameters
    ----------
    f: function
        the function to integrate: f(x, t, p=()). 
        The first argument is the real value(s) at this step, 
        the second is the time at this step, the third are a
        tuple of parameters of the system.
    x: numpy nd-array
        Array containing in the first position the initial 
        condition.
    t: numpy nd-array
        Array of the same lenght of x containing the time steps
        in wich the system will be evaluated.
    p: tuple
        Tuple of parameters of the function f.

    Result
    ------
    numpy nd-array
        Array x containing the results of the integration at 
        each time steps of t.
    """
    h = t[1]-t[0]
    step = len(t)
    for j in range(step-1):
        xj = x[j]
        tj = t[j]
        k1 = h * f( xj , tj , p )
        k2 = h * f( xj + k1 / 2 , tj + h / 2 , p)
        k3 = h * f( xj + k2 / 2 , tj + h / 2 , p)
        k4 = h * f( tj + h , xj + k3 , p)
        x[j+1] = xj + k1/6 + k2/3 + k3 / 3 + k4 / 6
    return x


@njit(fastmath = True)
def richardson_error(f, x, t, method = 'RK45', p=()):
    """
    Evaluate the richardson error for the integrator of
    this module.

    Parameters
    ----------
    f: function
        the function to integrate: f(x, t, p=()). 
        The first argument is the real value(s) at this step, 
        the second is the time at this step, the third are a
        tuple of parameters of the system.
    x: numpy nd-array
        Array containing in the first position the initial 
        condition.
    t: numpy nd-array
        Array of the same lenght of x containing the time steps
        in wich the system will be evaluated.
    method: string
        A string containing the name of the integration method
        for wich the Richardson error will be evaluated.
        The option are: 'RK45' (default), 'euler', 'midpoint',
        'trapezoid', 'AB'.
    p: tuple
        Tuple of parameters of the function f.

    Result
    ------
    numpy nd-array
        Array containing the error evaluated every two time 
        steps of t (so the lenght of the array is the half of x)
    """
    half_len = int(len(t)/2)
    t2 = t[:half_len] * 2

    if method == 'RK45':
        x_2t = RK45(f, x, t2, p)
        x_t = RK45(f, x, t, p)
        frac = 15
    if method == 'euler':
        x_2t = euler(f, x, t2, p)
        x_t = euler(f, x, t, p)
        frac = 1
    if method == 'trapezoid':
        x_2t = trapezoid(f, x, t2, p)
        x_t = trapezoid(f, x, t, p)
        frac = 3
    if method == 'midpoint':
        x_2t = midpoint(f, x, t2, p)
        x_t = midpoint(f, x, t, p)
        frac = 3
    if method == 'AB':
        x_2t = AB(f, x, t2, p)
        x_t = AB(f, x, t, p)
        frac = 3

    x_2t = x_2t[:half_len]
    x_t = x_t[::2]

    return np.abs(x_t - x_2t)/frac
