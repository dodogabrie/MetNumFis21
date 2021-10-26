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
from m4.PDE_tools import RKN, plot_evolution
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def FHS(f, x, nu, L, dx, der2):
    return - np.sin(10 * np.pi * x/L) * f + nu * simm_der2(f, dx, der2)

def burgers(f, x, nu, dx, der2):
    return - f * simm_der(f, dx, der2) + nu * simm_der2(f, dx, der2)

def main(Nt, dt, L, N, a, phi0, nu):
    tmax = dt * Nt
    x = np.linspace(0, L, N, endpoint = False)
    tt = np.arange(0, tmax, dt)
    dx = L/N
    alpha = nu * dt/dx**2 # Von Neumann stability
    print(f'dx = {dx}')
    print(f'Von Neumann factor: {alpha}')
    f0 = a * np.sin(2 * np.pi * x/L + phi0)
    der2 = np.empty(len(x))
    RKorder = 4
    ### Time evolution
    # Preparing the subplots
    fig = plt.figure(figsize = (15, 7))
    gs2 = GridSpec(2, 3)
    ax1 = fig.add_subplot(gs2[:, :-1])
    ax2 = fig.add_subplot(gs2[:-1, -1])
    ax3 = fig.add_subplot(gs2[-1, -1])

    f = np.copy(f0)
    # Siulation in time
    FHS_params = (x, nu, dx, der2,)
    meth_params = (4, burgers, dt)
    params = (*meth_params, FHS_params)
    plot_evolution(RKN, f, x, tt, params, ax = ax1, time_to_plot=5)
    info_init = rf'init = a*sin(2$\pi$ x/L)       x $\in \left[0, L\right]$     a = {a}'
    info_simulation = rf'      nu = {nu}     dt = {dt}       (L = {L}  ,  Nx = {N}) $\rightarrow$ dx = {dx:.3f}'
    info = info_init + info_simulation
    plt.suptitle(info, fontsize = 15)
    ax1.set_title('Solution in time')
    ax1.legend()
    # FFT
    fft_init = np.fft.rfft(f0)[1:-1]
    fft_final = np.fft.rfft(f)[1:-1]
    N = len(fft_final)
    freq = np.arange(1, N)/L
    ax2.set_title('Final spectrum')
    ax2.stem(freq, np.abs(fft_final[:-1])**2, 'b', \
             markerfmt=" ", basefmt="-b")
    ax3.set_title('Final spectrum (log scale)')
    ax3.plot(freq[1:], np.abs(fft_final[1:-1])**2)
    ax3.set_yscale('log')
#    ax2.plot(np.abs(fft_final))
    plt.tight_layout()
    plt.show()
#    plt.plot(freq[1:], np.abs(fft_final[1:-1])**2*freq[1:]**2)
#    plt.yscale('log')
#    plt.show()

    # Runge Kutta
#    FHS_params = (x, nu, L, dx, der2,)
#    meth_params = (4, FHS, dt)
#    param_func = (*meth_params, FHS_params)
#    aniplt.animated_with_slider(x, f0, RKN, Nt, dt, *param_func, dilat_size = 2)
    return

if __name__ == '__main__':
    # Parameters of the simulation
    Nt = 200      # Temporal steps
    dt = 1e-2     # Temporal step size
    Lx = 10        # Dimension of the x grid (in term of 2pi)
    Nx = 600       # Number of point in the grid
    # Parameters of the system.
    a  = 1
    phi0 = 0
    nu = 8*1e-3
    # Simulaion
    params = [Nt, dt, Lx, Nx, a, phi0, nu]
    main(*params)
