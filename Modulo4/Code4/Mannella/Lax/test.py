### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

import numpy as np
import time
from numba import njit
import LAX as lax
import m4.animated_plot as aniplt

def test_func(u):
    return np.exp( - u**2/2 ) * np.cos(2 * np.pi * u)

def test():

    v = -1
    dt = 0.002
    dx = 0.03
    Nt = 100
    n = 100
    ninner = 2

    alpha = v * dt /dx
    print(alpha)

    x = np.linspace(- n/2 * dx, n/2 * dx, n)
    t = np.linspace(0, dt * Nt, Nt)

    u = test_func(x)
    uinit = np.copy(u)

    aniplt.animated_full(lax.LAX, lax.LAX_complete, x, t, uinit, alpha, ninner, imported_title = 'Evoluzione Metodo LAX')
#   aniplt.animated_basic(x, uinit, lax.LAX, Nt, alpha, ninner) 


if __name__ == '__main__':
    test()
