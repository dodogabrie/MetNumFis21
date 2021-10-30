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
import m4.tridiag as td
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


def wiener_process_simple(u, D, f, x, dx, dt, ninner, t):
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

def diffusion_tridiag(u, N, alpha, ninner):
    """
    Implementation of the implicit solver of diffusive equation. The method is
    based on the inversion of the tridiagonal matrix of the system.
    """
    for i in range(ninner):
        diag_in = np.ones(N) * ( 1 + 2 * alpha )
        dlo_in = np.ones(N-1) * ( - alpha )
        dup_in = np.ones(N-1) * ( - alpha )
        u[1:-1], inv = td.solve(diag_in, dlo_in, dup_in, u[1:-1])
        u[0], u[-1] = u[-2], u[1]
    return u

def diffusion_tridiag_drift(u, drift, x, dt, N, alpha, beta, ninner, t):
    """
    Implementation of the implicit solver of diffusive equation with a drift
    term. The method is based on the inversion of the tridiagonal matrix of the
    system.
    """
    for i in range(ninner):
        t += dt
        diag_in = np.ones(N) * ( 1 + 2 * alpha )
        dlo_in = np.ones(N-1) * ( - alpha - beta * drift(x[1:], t))
        dup_in = np.ones(N-1) * ( - alpha + beta * drift(x[:-1], t))
        u, inv = td.solve(diag_in, dlo_in, dup_in, u)
        u[0], u[-1] = u[-2], u[1]
    return u


def test():
    def initial_value(x, t0, v):
        return np.exp( - (x + v*t0)**2/2 )
    def f(x, t):
        return - np.sin(t*5)*20 - t
    D = 1e-1
    dt = 2e-3
    dx = 9e-2
    Nt = 500
    n = 300
    ninner = 20

    print(f'Alpha : {D * dt / dx**2}')

    x = np.linspace(- n * dx, n * dx, 2*n)
    total_step = ninner*Nt
    t = np.linspace(0, dt * total_step, Nt)

    u = initial_value(x, 0, 1)
    uinit = np.copy(u)

    # Simple process
#    param_simple = (D, dx, dt, ninner)
#    plot_evolution(diffusion_simple, u, x, t, param_simple)

    # Wiener process simple
#    param_wiener_simple = [D, f, x, dx, dt, ninner, t[0]]
#    surface_xt(wiener_process_simple, uinit, x, t, param_wiener_simple, t_dependent=True, dilat_size = 0.1)
#    plot_evolution(wiener_process_simple, u, x, t, param_wiener_simple)
#    aniplt.animated_with_slider(wiener_process_simple, uinit, x, t,
#                                param_wiener_simple, plot_dim = None, t_dependent = True)
#    plot_evolution(wiener_process_simple, uinit, x, t, param_wiener_simple, t_dependent=True, time_to_plot=10)
#    surface_xt(wiener_process_simple, uinit, x, t, param_wiener_simple, t_dependent=True)

    # Tridiagonal Diffusion
#    alpha = dt/(dx**2) * D
#    N = int(len(x)-2)
#    param_diffusion_tridiag = [N, alpha, ninner]
#    aniplt.animated_with_slider(diffusion_tridiag, uinit, x, t,
#                                param_diffusion_tridiag, plot_dim = None)
#    plot_evolution(diffusion_tridiag, uinit, x, t, param_diffusion_tridiag, time_to_plot=10)
#    surface_xt(diffusion_tridiag, uinit, x, t, param_diffusion_tridiag)

    # Tridiagonal Diffusion with drift
    alpha = dt/(dx**2) * D
    beta = dt/(2*dx)
    N = len(x)
    param_diffusion_tridiag_drift = [f, x, dt, N, alpha, beta, ninner, t[0]]
#    aniplt.animated_with_slider(diffusion_tridiag, uinit, x, t,
#                                param_diffusion_tridiag_drift, plot_dim = None)
#    plot_evolution(diffusion_tridiag, uinit, x, t, param_diffusion_tridiag_drift, time_to_plot=10)
    surface_xt(diffusion_tridiag_drift, uinit, x, t, param_diffusion_tridiag_drift, t_dependent = True)


    plt.show()


    return

if __name__ == '__main__':
    test()
