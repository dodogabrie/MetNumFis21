"""
Lax Wendroff algorithm for a system like u__tt - c(x) * u_xx = 0 and for his
complication u_tt - c(x)_x * (c(x) * u)_x = 0
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

import matplotlib.pyplot as plt
import m4.animated_plot as aniplt
from m4.PDE_tools import surface_xt, RKN
import numpy as np
from numba import njit

#@njit(fastmath = True, cache = True)
def lax_wen_simple(u, c, dt, dx, ninner):
    """
    This function implement the Lax-Wendrom when c is a constant.
    """
    x, y = u[0], u[1]         # Array with component of u
    ca = c * dt/dx                 # Compute c * alpha
    for j in range(ninner):
        # LAX:
        xc = 0.5*( x[1:] + x[:-1] + ca * (y[1:] - y[:-1]) ) # This are shorter
        yc = 0.5*( y[1:] + y[:-1] + ca * (x[1:] - x[:-1]) ) # than x and y..
        # Stug. Leap Frog
        x[1:-1] +=  ca * (yc[1:] - yc[:-1])
        y[1:-1] +=  ca * (xc[1:] - xc[:-1])
        # Periodic conditions
        u[:, 0], u[:, -1] = u[:, -2], u[:, 1]
    return u

def lax_wen_hard(u, c, dt, dx, ninner):
    """
    This function implement the Lax-Wendrom method when c depends on x.
    """
    x, y = u[0], u[1]         # Array with component of u
    a = dt/dx                 # Compute alpha
    for j in range(ninner):
        # LAX:
        xc = 0.5*( x[1:] + x[:-1] + a * (c[1:] * y[1:] - c[:-1] * y[:-1]) ) # This are shorter
        yc = 0.5*( y[1:] + y[:-1] + a * (c[1:] * x[1:] - c[:-1] * x[:-1]) ) # than x and y..
        # Stug. Leap Frog
        x[1:-1] +=  a * (c[1:-1] * yc[1:] - c[:-2] * yc[:-1])
        y[1:-1] +=  a * (c[1:-1] * xc[1:] - c[:-2] * xc[:-1])
        # Periodic conditions
        u[:, 0], u[:, -1] = u[:, -2], u[:, 1]
    return u


def test():
    def initial_value(x, t0, v):
        return np.exp( - (x + v*t0)**2/2 )
    def sum_of_der(x, t0):
        return - 2 * (x + t0)*initial_value(x, t0)
    def cfunc(x):
        v0 = 1
        return v0 * np.exp(-x**2*0.01)
    s = 0
    dt = 0.05
    dx = 0.1
    Nt = 100
    n = 300
    c = 2
    ninner = 4
    print(f'Alpha = {dt/dx}')

    # Define the grid
    x = np.linspace(- n * dx, n * dx, 2*n)
    t = np.linspace(0, dt * Nt, Nt)

    # Fill u
    u = np.empty((2, len(x)))
    u[0] = initial_value(x, 0., c)
    u[1] = initial_value(x, 0., c)
    cx = cfunc(x)
    # Copy of u for plotting
    uinit = np.copy(u)
    lax_param_simple = (c, dt, dx, ninner)
    lax_param_hard = (cx, dt, dx, ninner)
#    aniplt.animated_full(lax_wen_simple, x, t, uinit, lax_param_simple, title = 'Evoluzione Metodo LAX-WENDROF', plot_dim = 0)
#    aniplt.animated_with_slider(lax_wen_simple, uinit, x, Nt, dt, lax_param_simple, plot_dim = 0, dilat_size = 0.1, title = None)
    surface_xt(lax_wen_hard, uinit, x, t, lax_param_hard, plot_dim = 0)

if __name__ == '__main__':
    test()

