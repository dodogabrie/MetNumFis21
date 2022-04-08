import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
from utils import plot_template, my_real_fft, evaluate_energy_density_spectrum

def analytic_sol(ampl_list, k_list, x, nu, T):
    sol = 0
    for A, k in zip(ampl_list, k_list):
        sol += A * np.exp( -nu * k**2 * T) * np.sin(k*x)
    return sol


def plot_diff_varying_derivative_nu(data_dir, analytic_sol = None):
    def plot_data(data, title, ax = None):
        if ax == None:
            fig, ax = plt.subplots(1,1, figsize = (8,8))
        ax.grid(alpha = 0.3)
        ax.minorticks_on()
        ax.tick_params('x', which='major', direction='in', length=5)
        ax.tick_params('y', which='major', direction='in', length=5)
        ax.tick_params('y', which='minor', direction='in', length=3, left=True)
        ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
        ax.set_xlabel('k', fontsize=15)
        ax.set_ylabel(r'$\left|u_k(t)\right|^2$')
        ax.set_yscale('log')
        ax.set_xscale('log')

        data = data.T
        x = data[0]
        dx = x[1]-x[0]
        L = x[-1] + dx

        u_init = data[1]

        mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx) 
        ax.scatter(k_u_init, mod_fft2_u_init, label = 'init', s = 10)
        ax.plot(k_u_init, mod_fft2_u_init, lw = 0)
         
        for u_t, nu in zip(data[2:], list_nu):
            mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t, dx)
#            list_m = (k_u / (2 * np.pi ) * L).astype(int)
            list_m = np.arange(1, 25)
            ax.scatter(k_u, mod_fft2_u, label = f'final', s = 6)
            if analytic_sol != None:
                ana_sol = analytic_sol([1 for m in list_m], k_u, x, nu, dt*N_step)
                mod_fft2_u, k_u = evaluate_energy_density_spectrum(ana_sol, dx)
                ax.plot(k_u, mod_fft2_u, lw = 0.8)
            ax.set_title(title)
        return ax

    dt = 0.1 # Temporal step
    N_step = 5000 # Number of Temporal steps

    filenames = ['dfs.txt', 'dfc.txt', 'fft.txt']
    n_files = len(filenames)

    all_data = [np.loadtxt(data_dir + filenames[i]) for i in range(n_files)]

    with open(data_dir + filenames[0]) as d:
        header = d.readlines()[0]
    list_nu = [float(nu) for nu in header.split('u_nu')[1:]]

    fig, axs = plt.subplots(3,1, figsize = (5,8))

    list_title = ['diff. fin. simm.', 'diff. fin. comp.', 'FFT']
    for ax, data, title in zip(axs, all_data, list_title):
        plot_data(data, title, ax)

    # Plot results
    plt.suptitle(r'$\partial_t u = \nu\partial^2_x u$ $\rightarrow$ RK2')
    plt.legend(fontsize = 9, frameon=False)
    plt.tight_layout()
#    plt.savefig('../figures/diffusion/diffusion_varying_der_new.png', dpi = 200)
    plt.show()
    return

if __name__ == '__main__':
    data_dir = '../../data/advection/varying_der/'
    plot_diff_varying_derivative_nu(data_dir, analytic_sol=None)
