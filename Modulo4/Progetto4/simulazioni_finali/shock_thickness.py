import sys
sys.path.append('../')

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import NumDerivative as der
import NumIntegrator as Int
from utils import plot_template, my_real_fft, my_real_ifft, evaluate_energy_density_spectrum
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit
from joblib import Parallel, delayed


# Define the right hand side term of the eq
def F_burger(u, nu, dx, der_method, der_method2):
    """
    Function of equation:
         du/dt = - u * du/dx = F(u)
    """
    return nu * der_method2(u, dx) - u * der_method(u, dx)

# Define the initial condition
def u_sin_simple(x, L, k = 1):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    return np.sin(k * 2*np.pi*x/L)

def template():
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('u', fontsize=15)
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)

    plt.rc('font', **{'size'   : 15})

    return fig, ax

def fit_tanh(x, A, l):
    L = 10
    x0 = L/2
    return -A*np.tanh((x-x0)/l)

def fit_shock_thickness(x, u):
    init = [1, 0.001] 
    idxmin = np.argmin(u)
    idxmax = np.argmax(u)
    if idxmin - idxmax < 10:
        idxmax -= 10
        idxmin += 10
    x = x[idxmax:idxmin]
    u = u[idxmax:idxmin]
    perc_cut = 90
    A_cut_up = perc_cut*u[0]/100
    A_cut_down = perc_cut*u[-1]/100
    mask_up = u < A_cut_up
    mask_down = u > A_cut_down
    mask = np.logical_and(mask_up, mask_down)
    x = x[mask]
    u = u[mask]
    pars, covm = curve_fit(fit_tanh, x, u, init, absolute_sigma=False)
    errors = np.sqrt(np.diag(covm))
    l, dl = pars[1], errors[1]
    A, dA = pars[0], errors[0]
    return l, dl, A, dA, u, x

def sim_varying_nu(i, n_simulation, x, u_init, N_step, int_method, 
        F, dt, order, nu, dx, d1, d2, save = False):
    u_t = np.copy(u_init) # create e copy to evolve it in time
    print(i, 'of ', n_simulation)
    u_t = np.copy(u_init) # create e copy to evolve it in time
    l0, _, _, _, _, _ = fit_shock_thickness(x, u_t)
    for j in range(N_step):# Evolution in time
        u_t = int_method(u_t,
                         F, dt, order, nu, dx, 
                         d1,  # first derivative
                         d2  # second derivative
                         )
        if (j+1)%int(N_step/100) == 0:
            print(f'{j/N_step*100:.0f}%   for process   {i}', end='\r')
            l, _, _, _, _, _ = fit_shock_thickness(x, u_t)
            if l < l0:
                l0 = l
            else: 
                break
    u_save = np.copy(u_t)
    u_save = np.append(u_save, nu)
    x_save = np.append(np.copy(x), -1)
    if save:
        np.savetxt(f'data_shock_thickness_parallel/u_t_{nu}.txt', np.column_stack((x_save, u_save)), header = '#x   u')
    return

def shock_thickness(int_method, order, u, F):
    L = 10 # Spatial size of the grid
    N = 10000 # spatial step of grid
    
    A = 1
    dt = 1e-5
    N_step = 500000 # Number of Temporal steps

    m = 1

    print('physic time:', dt * N_step)
    
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)

    u_init = A*u(x, L, k = m) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time

    d1 = der.simm_der
    d2 = der.simm_der2
    
    u_k2_init, _ = evaluate_energy_density_spectrum(u_init, dx)# FFT of init u 
    u_k = u_k2_init[m-1]
    
    k = 2*np.pi/L * m

    tau_nl = 1 / (k * u_k )
    _, ax = template()
    title = fr'L = {L},    N = {N},' + '\n' + f'dt = {dt:.1e},    N_step = {N_step:.1e}'
    plt.suptitle(title, fontsize = 15)
    ax.plot(x, u_init, label = 't=0')
    ax.scatter(x, u_init, s = 3)

    nu_list = np.linspace(0.001, 0.05, 16, endpoint=True)
    n_simulation = len(nu_list)
#    plot_nu_list = [0.001, 0.005]#[0.001, 0.003, 0.005]
    plot_nu_list = nu_list

    jobs = min(len(nu_list), 8)
    simulations = Parallel(n_jobs = jobs)(delayed(sim_varying_nu)(
        i, n_simulation, x, u_init, N_step, int_method, F, dt, order, nu, dx, d1, d2, save = True)
        for i, nu in enumerate(nu_list))

    mypath = './data_shock_thickness_parallel/'
    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for nu in plot_nu_list:
        for name in filenames:
            if f'{nu:.3f}' in name:
                print(f'{nu:.3f} already exists')
                data = np.loadtxt(mypath + name)
                x_from_save = data[:-1,0]
                u_from_save = data[:-1,1]
                ax.plot(x_from_save, u_from_save, label = f'nu = {nu}', lw = 0.7)
                ax.scatter(x, u_t.real, s = 5)
#    ax.set_xlim(4.98, 5.02)
#    plt.tight_layout()
    plt.legend()
#    plt.savefig(f'../figures/final/burger_shock_thickness.png', dpi = 150)
    plt.show()

def shock_thickness_analysis():
    L = 10
    mypath = './data_shock_thickness/'
    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    l_list = []
    dl_list = []
    nu_list = []
    for name in filenames:
        data = np.loadtxt(mypath + name)
        x = data[:-1,0]
        u = data[:-1,1]
        nu = data[-1, -1]
        l, dl, A, dA, u, x = fit_shock_thickness(x, u)
        print('nu = ',nu)
        print('l =', l)
        l_list.append(l)
        dl_list.append(dl)
        nu_list.append(nu)
        plt.scatter(x, u, s = 7)
        xx = np.linspace(x[0], x[-1], 1000)
        plt.plot(xx, fit_tanh(xx, A, l), lw = 0.7)
#        plt.plot(x, u)
#    plt.xlim(4.98, 5.02)
    plt.show()
    def lin_fit(x, m, q,):# k):
        return m * x  + q #+ k * x**2 
    init = [0.001, 0.001,]# 0.001]
    l_list = np.array(l_list)
    nu_list = np.array(nu_list)
    pars, covm = curve_fit(lin_fit, nu_list, l_list, init, absolute_sigma=False)
    errors = np.sqrt(np.diag(covm))
    A, dA = pars[0], errors[0]
    print(A)
    fig, axs = plt.subplots(2, 1)
    ax = axs[0]
    ax.set_xlabel(r'$\nu$', fontsize=15)
    ax.set_ylabel(r'$l$', fontsize=15)
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)

    title = fr'$\nu$ in funzione dello spessore dello shock $l$'
    plt.rc('font', **{'size'   : 15})
    plt.suptitle(title, fontsize = 15)

    ax.scatter(nu_list, l_list, label = 'dati', s = 10)
    xx = np.linspace(np.min(nu_list), np.max(nu_list), 1000)
    ax.plot(xx, lin_fit(xx, *pars), label = f'retta di fit, m = {pars[0]:.2f}')
    ax = axs[1]
    ax.scatter(nu_list, l_list-lin_fit(nu_list, *pars))
    ax = axs[0]
    ax.legend()
#    plt.savefig(f'../figures/final/thickness_fit.png', dpi = 150)
    plt.show()
    return
 

if __name__ == '__main__':
#    shock_thickness(Int.RKN, 4, u_sin_simple, F_burger)
    shock_thickness_analysis()
