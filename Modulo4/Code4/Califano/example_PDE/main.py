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

    #------------- Initialize the problem -------------------------------------
    tmax = dt * Nt # Total time
    x = np.linspace(0, L, N, endpoint = False) # grid of the problem
    tt = np.arange(0, tmax, dt)                # Temporal grid of the problem
    dx = L/N                                   # spatial step size
    alpha = nu * dt/dx**2                      # Von Neumann stability
    print(f'dx = {dx}')
    print(f'Von Neumann factor: {alpha}')
    f0 = a * np.sin(2 * np.pi * x/L + phi0)    # Initial condition
    der2 = np.empty(len(x))                    # Empty array for smart deriv.
    RKorder = 4                                # RK order
    animation = False                          # Plot an animation in time
    #--------------------------------------------------------------------------

    #----------- Preparing the plots ------------------------------------------
    fig = plt.figure(figsize = (16, 8))
    gs2 = GridSpec(2, 3)
    ax1 = fig.add_subplot(gs2[:, :-1])
    ax2 = fig.add_subplot(gs2[:-1, -1])
    ax3 = fig.add_subplot(gs2[-1, -1])
    #--------------------------------------------------------------------------

    #------------ Time evolution ----------------------------------------------
    f = np.copy(f0)                   # Save the initial condition apart
    FHS_params = (x, nu, dx, der2,)   # Parameters of the Right side
    meth_params = (RKorder, burgers, dt)    # Parameters of RK method
    params = (*meth_params, FHS_params) # All the parameters together
    # Evolution of the system direcly plotted
    plot_evolution(RKN, f, x, tt, params, ax = ax1, time_to_plot=5)
    #--------------------------------------------------------------------------

    # Title of plots containing info on simulation
    info_init = rf'init = a*sin(2$\pi$ x/L)       x $\in \left[0, L\right]$     a = {a}     (L = {L}  ,  Nx = {N}) $\rightarrow$ dx = {dx:.3f}'
    info_simulation = rf'      nu = {nu}     dt = {dt}       ,   $\alpha$ = {alpha:.2f}'
    info = info_init + '\n' + info_simulation
    plt.suptitle(info, fontsize = 17)
    ax1.set_title('Solution in time', fontsize = 14)
    ax1.set_xlabel('x', fontsize = 14)
    ax1.set_ylabel('u(t)', fontsize = 14)
    ax1.set_xlim(np.min(x), np.max(x))
    ax1.legend(fontsize=13)

    #------------- Fourier transform ------------------------------------------
    fft_init = np.fft.rfft(f0)[1:-1]
    fft_final = np.fft.rfft(f)[1:-1]
    N = len(fft_final)
    freq = np.arange(1, N)/L

    # Stem plot
    ax2.set_title('Final spectrum (log,lin)', fontsize = 14)
    ax2.scatter(freq[1:], np.abs(fft_final[1:-1])**2, color = 'black', s = 8)
    ax2.set_yscale('log')
    ax2.set_xlabel('k', fontsize = 14)
    ax2.set_ylabel('FFT count', fontsize = 12)
    ax2.set_xlim(-0.1, np.max(freq))

    # Logaritmic plot + check behaviur
    ax3.set_title('Final spectrum (log, log)', fontsize = 14)
    ax3.scatter(freq[1:], np.abs(fft_final[1:-1])**2, label = 'FFT', s = 8)
    ax3.scatter(freq[1:], np.abs(fft_final[1:-1])**2*freq[1:]**2, s = 8,
             label = 'FFT * 1/k^2')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_xlim(np.min(freq[1:]), np.max(freq[1:]))
    ax3.set_ylim(1e-6, 1e4)
    ax3.set_xlabel('k', fontsize = 14)
    ax3.set_ylabel('FFT count', fontsize = 12)
    ax3.legend(fontsize=13)
    #--------------------------------------------------------------------------

    # Show results
    plt.tight_layout()
    plt.show()

    #---------------- Animation -----------------------------------------------
    if animation:
        FHS_params = (x, nu, dx, der2,)   # Parameters of the Right side
        meth_params = (4, burgers, dt)
        param_func = (*meth_params, FHS_params)
        aniplt.animated_with_slider(RKN, f0, x, tt, param_func, dilat_size = 0.1)
    #--------------------------------------------------------------------------
    return

if __name__ == '__main__':
    # Parameters of the simulation
    Nt = 150      # Temporal steps
    dt = 2e-2     # Temporal step size
    Lx = 11        # Dimension of the x grid (in term of 2pi)
    Nx = 600       # Number of point in the grid
    # Parameters of the system.
    a  = 1
    phi0 = 0
    nu = 8*1e-3
    # Simulaion
    params = [Nt, dt, Lx, Nx, a, phi0, nu]
    main(*params)
