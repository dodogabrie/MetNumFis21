"""
This file solve the PDE of a system of gravitational waves using the 
Lax-Wendron method.
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

import matplotlib.pyplot as plt
import m4.animated_plot as aniplt
from m4.PDE_tools import surface_xt
import numpy as np
from numba import njit

#@njit(fastmath = True, cache = True)
def lax_wen_complete(evo, u, Nt, gridx, v, s, dt, dx, ninner):
    """
    This funciton fill the meshgrid (space, time) of the evolution of the 
    system.
    """
    evo[0] = u[0]
    for i in range(Nt-1):
        u = lax_wen(u, gridx, v, s, dt, dx, ninner)
        evo[i+1] = u[0]
    return evo

#@njit(fastmath = True, cache = True)
def lax_wen(u, gridx, v, s, dt, dx, ninner, Nt = 2):
    """
    This function implement the Lax-Wendrom method using numpy slicing.
    It is probably efficient but I don't know, for sure is elegant!
    """
    h1 = lambda u, xx : v(xx) * u + s # Function for the second component of h
    x, y = u[0], u[1]         # Array with component of u
    xc, yc = np.copy(u)       # Aux copy of the components
    # Loop over time
    for i in range(Nt-1):     # By default this is bypassed
        # Loop to speed up the wave evolution (skip some temporal step)
        for j in range(ninner):
            # LAX:    |--------------REGULAR--------------------| |-----------ADDITIVE TERM---------------------------------------|  
            xc[:-1] = ( x[1:] + x[:-1] - dt/dx * (x[1:] - x[:-1]) + dt/2. * (        y[1:]           +         y[:-1]           ) )/2.
            yc[:-1] = ( y[1:] + y[:-1] + dt/dx * (y[1:] - y[:-1]) + dt/2. * ( h1( x[1:], gridx[1:] ) + h1( x[:-1], gridx[:-1])   ) )/2.
            # Stug. Leap Frog                         |---------------ADDITIVE TERM------------------------------------------|
            x[1:-1] += - dt/dx * (xc[1:-1] - xc[:-2]) + dt/2. * (           yc[1:-1]           +           yc[:-2]    ) 
            y[1:-1] +=   dt/dx * (yc[1:-1] - yc[:-2]) + dt/2. * ( h1( xc[1:-1] , gridx[1:-1] ) + h1( xc[:-2] , gridx[:-2] ) ) 
            # Periodic conditions
            u[:, 0], u[:, -1] = u[:, -2], u[:, 1]
    return u

def test():
    def initial_value(x, t0): 
        return np.exp( - (x - t0)**2/2 )
    def sum_of_der(x, t0):
        return 0#- 2 * (x + t0)*initial_value(x, t0) 
    def v(x):
        return - np.exp(-(x-10)**2)
    def v1(x):
        v0 = 1
        return - v0 * np.exp(-x**2*0.01)
    s = 0
    dt = 0.05
    dx = 0.1
    Nt = 500
    n = 300
    ninner = 2

    # Define the grid
    x = np.linspace(- n * dx, n * dx, 2*n)
    t = np.linspace(0, dt * Nt, Nt)

    # Fill u
    u = np.empty((2, len(x)))
    u[0] = initial_value(x, 0.)
    u[1] = sum_of_der(x, 0.)
    # Copy of u for plotting
    uinit = np.copy(u)
    lax_param = (x, v1, s, dt, dx, ninner)
#    aniplt.animated_full(lax_wen, lax_wen_complete, x, t, uinit, *lax_param, imported_title = 'Evoluzione Metodo LAX-WENDROF', plot_dim = 0)
    aniplt.animated_with_slider(x, uinit, lax_wen, Nt, dt, *lax_param, plot_dim = 0, dilat_size = 0.1, imported_title = None)
#    surface_xt(lax_wen, x, t, uinit, lax_param, plot_dim = 0)
    

if __name__ == '__main__':
    test()
                           
