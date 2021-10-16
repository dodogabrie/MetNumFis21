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
import numpy as np
from numba import njit

#@njit(fastmath = True, cache = True)
def lax_wen_complete(evo, u, Nt, v, s, dt, dx, ninner):
    """
    This funciton fill the meshgrid (space, time) of the evolution of the 
    system.
    """
    evo[0] = u[0]
    for i in range(Nt-1):
        u = lax_wen(u, v, s, dt, dx, ninner)
        evo[i+1] = u[0]
    return evo

#@njit(fastmath = True, cache = True)
def lax_wen(u, v, s, dt, dx, ninner, Nt = 2):
    """
    This function implement the Lax-Wendrom method using numpy slicing.
    It is probably efficient but I don't know, for sure is elegant!
    """
    h1 = lambda u : v * u + s # Function for the second component of h
    x, y = u[0], u[1]         # Array with component of u
    xc, yc = np.copy(u)       # Aux copy of the components
    # Loop over time
    for i in range(Nt-1):     # By default this is bypassed
        # Loop to speed up the wave evolution (skip some temporal step)
        for j in range(ninner):
            # LAX:    |--------------REGULAR--------------------| |-----------ADDITIVE TERM--------------|  
            xc[:-1] = ( x[1:] + x[:-1] - dt/dx * (x[1:] - x[:-1]) + dt/2. * (     y[1:]   +    y[:-1]    ) )/2.
            yc[:-1] = ( y[1:] + y[:-1] + dt/dx * (y[1:] - y[:-1]) + dt/2. * ( h1( x[1:] ) + h1( x[:-1] ) ) )/2.
            # Stug. Leap Frog                         |---------------ADDITIVE TERM--------------|
            x[1:-1] += - dt/dx * (xc[1:-1] - xc[:-2]) + dt/2. * (   yc[1:-1]     +    yc[:-2]    ) 
            y[1:-1] +=   dt/dx * (yc[1:-1] - yc[:-2]) + dt/2. * ( h1( xc[1:-1] ) + h1( xc[:-2] ) ) 
            # Periodic conditions
            u[:, 0], u[:, -1] = u[:, -2], u[:, 1]
    return u

def test():
    def initial_value(x, t0): 
        return np.exp( - (x + t0)**2/2 )
    def sum_of_der(x, t0):
        return - 2 * (x + t0)*initial_value(x, t0) 
    v = 0
    s = 0
    dt = 0.05
    dx = 0.1
    Nt = 100
    n = 100
    ninner = 3

    # Define the grid
    x = np.linspace(- n * dx, n * dx, 2*n)
    t = np.linspace(0, dt * Nt, Nt)

    # Fill u
    u = np.empty((2, len(x)))
    u[0] = initial_value(x, 0.)
    u[1] = sum_of_der(x, 0.)
    # Copy of u for plotting
    uinit = np.copy(u)

    aniplt.animated_full(lax_wen, lax_wen_complete, x, t, uinit, v, s, dt, dx, ninner, imported_title = 'Evoluzione Metodo LAX-WENDROF', plot_dim = 0)


if __name__ == '__main__':
    test()
                           
