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
import LAX_WEN as lax_wen
import m4.animated_plot as aniplt

def test_func(u):
    return np.exp( - u**2/2 ) * np.cos(2 * np.pi * u)

def test():

    v = -1
    dt = 0.0999
    dx = 0.1
    Nt = 200
    n = 200
    ninner = 1

    alpha = v * dt /dx
    print(alpha)

    x = np.linspace(- n/2 * dx, n/2 * dx, n)
    t = np.linspace(0, dt * Nt, Nt)

    u = test_func(x)
    uinit = np.copy(u)
    uinit1 = np.copy(u)

#    aniplt.animated_full(lax_wen.LAX_WEN, uinit1, x, t, (alpha, ninner), title = 'Evoluzione Metodo LAX-WENDROF')
    aniplt.animated_with_slider(lax_wen.LAX_WEN, uinit1, x, t, (alpha, ninner), title = 'Evoluzione Metodo LAX-WENDROF')


if __name__ == '__main__':
    test()
