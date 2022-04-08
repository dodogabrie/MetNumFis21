"""
This file uses some integrators to solve the burger equation:
    d_t ( u ) = nu * d_x^2 ( u ) - u * d_x ( F ( u ) )
Where d_t and d_x are the derivative respect t and x.
"""
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

def gaussian(x, mu, sigma):
    return 1/(np.sqrt(2 * np.pi * sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))
    
def ampl2(int_method, order, u, F, der_method = 'dfs', intermediate_step = 1):
    """
    This function plot the square of the Fourier coefficient of u 
    at the initial time and after few step.
    The bigger k are the first to explode.

    Parameters
    ----------
    int_method: function 
        Integration method (euler or RK2).
    u: function
        Initial condition
    F: function
        Right hand side
    """
    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 1000 # spatial step of grid
    
    case1 = 1 # t_nl ~ t_d
    case2 = 0 # t_nl << t_d

    if case1:
        A = 1e-3
        nu = 0.1 # diffusive parameter
        dt = 0.0005
        N_step = 6000 # Number of Temporal steps
    elif case2:
        A = 1e-3
        nu = 0.000005 # diffusive parameter
        dt = 0.05 # Temporal step
        N_step = 9000 # Number of Temporal steps
    else:
        A = 1
        nu = 0.0055 # diffusive parameter
        dt = 0.005 
        N_step = 3*65500 # Number of Temporal steps

    print('physic time:', dt * N_step)
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)

    # Define starting function
#    u_init = u(x, L) # save initial condition 
    m = 5
    u_init = A*u(x, L, k = m) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time

#    von_neumann = nu * dt /(dx**2)

    if der_method == 'fft':
        d1 = der.fft_der
        d2 = der.fft_der2
    elif der_method == 'dfs':
        d1 = der.simm_der
        d2 = der.simm_der2
    elif der_method == 'dfc':
        d1 = der.diff_fin_comp_der
        d2 = der.diff_fin_comp_der2
    else: 
        d1 = der.fft_der
        d2 = der.fft_der2
    
    u_k2_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx)# FFT of initial u 
    u_k = u_k2_init[m-1]
    last_u_k = u_k2_init[-1]
    last_k = k_u_init[-1]
    
    k = 2*np.pi/L * m
    print(nu)
    tau_d = 1 / ( k**2 * nu )
    tau_nl = 1 / (k * u_k )
    print('tau_d:', tau_d)
    print('tau_nl:', tau_nl)


    title = fr'L = {L},    N = {N},    dt = {dt:.1e}    $\nu$ = {nu},' + '\n' + fr'N_step = {N_step:.0e},    $\tau_o$ =  {tau_nl:.2f},    $\tau_d$ =  {tau_d:.2f}   '
    fig, axs = plot_template(2,1, figsize = (6, 9))
    plt.rc('font', **{'size'   : 15})
    plt.suptitle(title, fontsize = 15)

    ax = axs[0]
    ax.plot(x, u_init, label = 't=0')
    ax.scatter(x, u_init, s = 3)

    ax = axs[1]
    ax.scatter(k_u_init, u_k2_init, label = 't=0', s=8)

    for j in range(intermediate_step):
        for i in range(int(N_step/intermediate_step)):# Evolution in time
            u_t = int_method(u_t, F, dt, order, nu, dx, 
                             d1,  # first derivative
                             d2  # second derivative
                             )
        tt = (j+1)*int(N_step/intermediate_step)
        u_k2, k_u = evaluate_energy_density_spectrum(u_t.real, dx) # FFT of final u 
        ax = axs[0]
        ax.plot(x, u_t, label = f't={dt*tt}', lw = 0.7)
        ax.scatter(x, u_t.real, s = 5)
        ax = axs[1]
        ax.scatter(k_u, u_k2, label = f't={dt*tt}', s = 8)
    
#    plt.tight_layout()
    plt.legend()
    #plt.savefig(f'../figures/final/burger_t_nl_sim_t_d.png', dpi = 150)
    plt.show()

def shock_thickness(int_method, order, u, F, der_method = 'dfs'):
    """
    This function plot the square of the Fourier coefficient of u 
    at the initial time and after few step.
    The bigger k are the first to explode.

    Parameters
    ----------
    int_method: function 
        Integration method (euler or RK2).
    u: function
        Initial condition
    F: function
        Right hand side
    """
    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 10000 # spatial step of grid
    
    A = 1
    dt = 1e-5
    N_step = 500000 # Number of Temporal steps

    print('physic time:', dt * N_step)
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)

    # Define starting function
#    u_init = u(x, L) # save initial condition 
    m = 1
    u_init = A*u(x, L, k = m) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time

#    von_neumann = nu * dt /(dx**2)

    if der_method == 'fft':
        d1 = der.fft_der
        d2 = der.fft_der2
    elif der_method == 'dfs':
        d1 = der.simm_der
        d2 = der.simm_der2
    elif der_method == 'dfc':
        d1 = der.diff_fin_comp_der
        d2 = der.diff_fin_comp_der2
    else: 
        d1 = der.fft_der
        d2 = der.fft_der2
    
    u_k2_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx)# FFT of initial u 
    u_k = u_k2_init[m-1]
    last_u_k = u_k2_init[-1]
    last_k = k_u_init[-1]
    
    k = 2*np.pi/L * m

    tau_nl = 1 / (k * u_k )
    title = fr'L = {L},    N = {N},' + '\n' + f'dt = {dt:.1e},    N_step = {N_step:.1e}'
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
    plt.suptitle(title, fontsize = 15)
    ax.plot(x, u_init, label = 't=0')
    ax.scatter(x, u_init, s = 3)

    step_size = 0.001
    nu_list = np.arange(0.001, 0.05 + step_size, step_size)[-2:]
    n_simulation = len(nu_list)
    plot_nu_list = [0.001, 0.005]#[0.001, 0.003, 0.005]
    plot_nu_list = nu_list
    idxmin = np.argmin(u_t)
    idxmax = np.argmax(u_t)
    n_pt_old = idxmin - idxmax + 1
    n_pt_save = n_pt_old
    print('mid point:', n_pt_old)
    for i, nu in enumerate(nu_list):
        print(i, 'of ', n_simulation)
        tau_d = 1 / ( k**2 * nu )
#        print('tau_d:', tau_d)
#        print('tau_nl:', tau_nl)
        n_pt_old = n_pt_save
        u_t = np.copy(u_init) # create e copy to evolve it in time
        for i in range(N_step):# Evolution in time
            if i%1000 == 0: print(i, end='\r')
            u_t = int_method(u_t, F, dt, order, nu, dx, 
                             d1,  # first derivative
                             d2  # second derivative
                             )
            idxmin = np.argmin(u_t)
            idxmax = np.argmax(u_t)
            n_pt = idxmin - idxmax
            if n_pt_old > n_pt:
                n_pt_old = n_pt
            else: 
                print('stopped at', i)
                break
        u_save = np.copy(u_t)
        u_save = np.append(u_save, nu)
        x_save = np.append(np.copy(x), -1)
#        np.savetxt(f'data_shock_thickness/u_t_{nu}.txt', np.column_stack((x_save, u_save)), header = '#x   u')
        if nu in plot_nu_list:
            ax.plot(x, u_t, label = f'nu = {nu}', lw = 0.7)
            ax.scatter(x, u_t.real, s = 5)
#    ax.set_xlim(4.98, 5.02)
#    plt.tight_layout()
    plt.legend()
#    plt.savefig(f'../figures/final/burger_shock_thickness.png', dpi = 150)
    plt.show()

def fit_shock_thickness():
    L = 10
    def fit_tanh(x, A, l):
        x0 = L/2
        return -A*np.tanh((x-x0)/l)
    mypath = './data_shock_thickness/'
    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    l_list = []
    A_list = []
    dl_list = []
    nu_list = []
    for name in filenames:
        init = [1, 0.001] 
        data = np.loadtxt(mypath + name)
        x = data[:-1,0]
        u = data[:-1,1]
        idxmin = np.argmin(u)
        idxmax = np.argmax(u)
        if idxmin - idxmax < 10:
            idxmax -= 10
            idxmin += 10
        x = x[idxmax:idxmin]
        u = u[idxmax:idxmin]
        nu = data[-1, -1]
        print('nu = ',nu)
        pars, covm = curve_fit(fit_tanh, x, u, init, absolute_sigma=False)
        errors = np.sqrt(np.diag(covm))
        l, dl = pars[1], errors[1]
        A, dA = pars[0], errors[0]
        print('l =', l)
        l_list.append(l)
        dl_list.append(dl)
        nu_list.append(nu)
        A_list.append(A)
        plt.scatter(x, u, s = 7)
        xx = np.linspace(x[0], x[-1], 1000)
        plt.plot(xx, fit_tanh(xx, *pars), lw = 0.7)
#        plt.plot(x, u)
#    plt.xlim(4.98, 5.02)
    plt.show()
    def lin_fit(x, m, q):
        return m * x + q 
    init = [0.001, 0.001,]# 0.001]
    n_data = len(l_list)
    l_list = np.array(l_list)
    A_list = np.array(A_list)
    nu_list = np.array(nu_list)
    x = nu_list/A_list
    pars, covm = curve_fit(lin_fit, x, l_list, init, absolute_sigma=False)
    errors = np.sqrt(np.diag(covm))
    fig, ax = plt.subplots(1, 1, figsize = (6, 4.5))
#    ax = axs[0]
    ax.set_xlabel(r'$\nu/A$', fontsize=15)
    ax.set_ylabel(r'$l$', fontsize=15)
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)

    title = fr'Spessore dello shock $l$ in funzione di $\nu$'
    plt.rc('font', **{'size'   : 15})
    plt.suptitle(title, fontsize = 15)

    ax.scatter(x, l_list, label = 'dati', s = 14, alpha = 0.8)
    xx = np.linspace(np.min(x), np.max(x), 1000)
    ax.plot(xx, lin_fit(xx, *pars), label = f'retta di fit, m = {pars[0]:.2f}', color = 'tab:red', lw = 0.8)
#    ax = axs[1]
#    ax.scatter(nu_list, l_list-lin_fit(nu_list, *pars))
#    ax = axs[0]
    ax.legend()
    plt.savefig(f'../figures/final/thickness_fit.png', dpi = 150)
    plt.show()
    return
 

if __name__ == '__main__':
    der_method = 'dfs'
#    shock_thickness(Int.RKN, 4, u_sin_simple, F_burger, der_method)
    fit_shock_thickness()
