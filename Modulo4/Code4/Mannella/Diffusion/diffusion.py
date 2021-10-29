"""
Methods to solve the diffusion equation.
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

import matplotlib.pyplot as plt
import m4.animated_plot as aniplt
from m4.PDE_tools import surface_xt, plot_evolution
import numpy as np
from numba import njit

def diffusion_simple(u, D, dx, dt, ninner):
    """
    A simple impleentation of the solution of diffusive equation.
    """
    alpha = dt/(dx**2) * D
    for i in range(ninner):
        u[1:-1] += alpha * (u[2:] -2 * u[1:-1] + u[:-2])
        u[0], u[-1] = u[-2], u[1]
    return u

def wiener_process(u, D, f, x, dx, dt, ninner, t):
    """
    Solution to the Wiener Process:
        dx = fdt + sqrt(2D) * dw
    Nuerically this became:
        d_t(P) = - d_x[ f(x, t) ] * P + D d_x^2(P)
    And the solution is just the diffusion simple minus one term.
    """
    alpha = dt/(dx**2) * D
    cfl2 = dt/(2*dx)
    for i in range(ninner):
        u[1:-1] += ( alpha * ( u[2:] - 2 * u[1:-1] + u[:-2])
                    - cfl2 * ( f(x[2:], t) * u[2:] - f(x[:-2], t) * u[:-2]) )
        u[0], u[-1] = u[-2], u[1]
        t += dt
    return u

def test():
    def initial_value(x, t0, v):
        return np.exp( - (x + v*t0)**2/2 )
    def f(x, t):
        return - 10*np.cos(2*np.pi*x) - t*7
    D = 9.9e-1
    dt = 2e-3
    dx = 9e-2
    Nt = 100
    n = 200
    ninner = 20

    print(f'Alpha : {D * dt / dx**2}')

    x = np.linspace(- n * dx, n * dx, 2*n)
    total_step = ninner*Nt
    t = np.linspace(0, dt * total_step, Nt)

    u = initial_value(x, 0, 1)
    uinit = np.copy(u)
    param_simple = (D, dx, dt, ninner)
    param_wiener = [D, f, x, dx, dt, ninner, t[0]]

    # Simple process
#    plot_evolution(diffusion_simple, u, x, t, param_simple)

    # Wiener process
    surface_xt(wiener_process, uinit, x, t, param_wiener, t_dependent=True, dilat_size = 0.1)
#    plot_evolution(wiener_process, u, x, t, param_wiener)
#    aniplt.animated_with_slider(wiener_process, uinit, x, t, param_wiener, plot_dim = None, t_dependent = True)
#    plot_evolution(wiener_process, uinit, x, t, param_wiener, t_dependent=True, time_to_plot=10)
#    surface_xt(wiener_process, uinit, x, t, param_wiener, t_dependent=True)

    plt.show()


    return

if __name__ == '__main__':
    test()
