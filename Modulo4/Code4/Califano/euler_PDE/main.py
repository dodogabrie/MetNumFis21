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

def solver(f, x, dt, dx, der2):
    f[:] = f + (- np.sin(x) * f + simm_der2(f, dx, der2)) * dt 
    return f

def main(Nt = 100, dt=1e-2, alpha = 0.5, L=10, a = 1e-2, phi0 = 0):
    dx = np.sqrt(dt/alpha) 
    x = np.arange(0, L, dx)
    print(f'Number of points = {len(x)}')
    f0 = a * np.sin(2 * np.pi * x/L + phi0)
    der2 = np.empty(len(x))
    aniplt.animated_basic(x, f0, solver, Nt, x, dt, dx, der2)
    return

if __name__ == '__main__':
    main()
