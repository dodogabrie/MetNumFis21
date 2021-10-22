### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

import numpy as np
from numba import njit
from m4.derivate import simm_der, simm_der2, shift_test
import m4.animated_plot as aniplt
import matplotlib.pyplot as plt


def FHS(f, x, nu, L, dx, der2):
    return - np.sin(10 * np.pi * x/L) * f + nu * simm_der2(f, dx, der2)

def euler(f, FHS, dt, param):
    f[:] = f + FHS(f, *param) * dt 
    return f

def RKN(f, FHS, dt, param, N):
    g = np.copy(f)
    for k in range(N,0,-1):
        g = f + 1/k * FHS(g, *param) * dt
    f[:] = g
    return f

def main(Nt, dt, L, N, a, phi0, nu):
    x = np.linspace(0, L, N, endpoint = False)
    dx = L/N
    alpha = nu * dt/dx**2 # Von Neumann stability
    print(f'dx = {dx}')
    print(f'Von Neumann factor: {alpha}')
    f0 = a * np.sin(2 * np.pi * x/L + phi0)
    der2 = np.empty(len(x))
    RKorder = 4
    # Euler
#    param_func = (FHS, dt, (x, nu, L, dx, der2,))
#    aniplt.animated_with_slider(x, f0, euler, Nt, dt, *param_func, dilat_size = 2)
    # Runge Kutta
    param_func = (FHS, dt, (x, nu, L, dx, der2,),4,)
    aniplt.animated_with_slider(x, f0, RKN, Nt, dt, *param_func, dilat_size = 2)
    # Testing diffusion and transport of wave
#    param_func = (test_diffusion, dt, (x, nu, L, dx, der2,),4,)
#    aniplt.animated_with_slider(x, f0, RKN, Nt, dt, *param_func)
#    param_func = (test_wave, dt, (dx, der2,), 4,)
#    aniplt.animated_with_slider(x, f0, RKN, Nt, dt, *param_func)
    return

def test_diffusion(f, x, nu, L, dx, der2):
    return nu * simm_der2(f, dx, der2)

def test_wave(f, dx, der2):
    return - simm_der(f, dx, der2)


if __name__ == '__main__':
    # Parameters of the simulation
    Nt = 100      # Temporal steps
    dt = 3e-2     # Temporal step size
    Lx = 10        # Dimension of the x grid (in term of 2pi)
    Nx = 60       # Number of point in the grid
    # Parameters of the system.
    a  = 1e-2
    phi0 = 0
    nu = 0.3
    # Simulaion
    params = [Nt, dt, Lx, Nx, a, phi0, nu]
    main(*params)
