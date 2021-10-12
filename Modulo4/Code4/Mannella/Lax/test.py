import numpy as np
import time
from numba import njit
import LAX as lax
import plot_wave

def test_func(u):
    return np.exp( - u**2/2 ) * np.cos(2 * np.pi * u)

def test():

    v = -1
    dt = 0.002
    dx = 0.03
    Nt = 100
    n = 50
    ninner = 2

    alpha = v * dt /dx
    print(alpha)

    x = np.linspace(- n/2 * dx, n/2 * dx, n)
    t = np.linspace(0, dt * Nt, Nt)

    u = test_func(x)
    uinit = np.copy(u)


    plot_wave.animated_full(lax.LAX, lax.LAX_complete, x, t, uinit, alpha, ninner, imported_title = 'Evoluzione Metodo LAX')
#    plot_wave.animated_basic(x, uinit, lax.LAX, Nt, alpha, ninner) 


if __name__ == '__main__':
    test()
