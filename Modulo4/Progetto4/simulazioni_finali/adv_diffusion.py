import sys

from scipy.fftpack.basic import fft
sys.path.append('../')

import numpy as np 
import matplotlib.pyplot as plt
import NumDerivative as der
import NumIntegrator as Int
import NumScheme as NS
from scipy.optimize import curve_fit
from utils import my_real_fft, plot_template, evaluate_energy_density_spectrum
from utils import check_CFL, check_VonNeumann
from utils import analytic_sol_u_sin_simple_advection, analytic_sol_gaussian_advection
from utils import analytic_sol_u_sin_simple_diffusion
from utils import u_sin_simple, gaussian
from utils import advection_diffusion
from utils import plot_results

def initialization(L, N, u, init_params):
    dx = L/N
    x = np.linspace(0,L,N, endpoint = False)

    check_CFL(c, dx, dt, print_info = False) # check CFL condition
    check_VonNeumann(nu, dt, dx, print_info = False) # Check Von Neumann on diffusion only

    # Define starting function
    u_init = u(x, init_params) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
    return dx, x, u_init, u_t

def evolve_advection(L, N, nu, dt, N_step, c,
                     u, F, der1, der2, int_method, RKorder, 
                     init_params, u_prev = []):
    dx, x, u_init, u_t = initialization(L, N, u, init_params)
    if len(u_prev) != 0:
        u_t = u_prev
    for _ in range(N_step): # temporal evolution
            u_t = int_method(u_t, F, dt, RKorder, c, nu, dx, der1, der2) 
            u_k2, _ = evaluate_energy_density_spectrum(u_t, dx)
    return u_init, u_t, x, dx

def check_derivative_effect(L, N, nu, dt, N_step, c,
                            u, F, list_der1, list_der2, name_der, int_method,
                            RKorder, init_params, intermediate_step = 1, return_max = False):
    CFL = c*dt/L * N
    title = f'L = {L},    N = {N},    dt = {dt:.1e},   nu = {nu:.0e}\n N_step = {N_step:.0e},    c = {c},    CFL = {CFL:.2f}'
#    fig, axs = plot_template(2,1, figsize = (6, 9))
#    plt.rc('font', **{'size'   : 15})
#    plt.suptitle(title, fontsize = 15)
    if intermediate_step != 1:
        N_step = int(N_step/intermediate_step)
    dx, x, u_init, u_t = initialization(L, N, u, init_params)
    max_list = np.empty((intermediate_step, int(N/2) - 1))
#    axs = plot_results(u_t, x, dx, label = f't=0', axs = axs)
    for i, (der1, der2, name) in enumerate(zip(list_der1, list_der2, name_der)):
        for i in range(intermediate_step):
            print(dt*N_step*(i+1))
            results = evolve_advection(L, N, nu, dt, N_step, c,
                                       u, F, der1, der2, int_method, 
                                       RKorder, init_params, u_prev=u_t)
            u_init, u_t, x, dx = results
            u_k, k = evaluate_energy_density_spectrum(u_t, dx)
            max_list[i] = u_k
    #        u_exact = analytic_sol_u_sin_simple_advection(x, m, c, N_step*dt)
#            axs = plot_results(u_t, x, dx, label = f't={dt*N_step*(i+1):.0f}', axs = axs)
#    axs[0].legend(loc='best') 
#    axs[1].legend(loc='center left') 
#    plt.savefig(f'../figures/final/advection_CFL{CFL:.2f}_gaussian.png', dpi = 150)
#    plt.tight_layout()
#    plt.show()
    if return_max: return np.array(max_list)
    else: return

def fit_Max_gaussian(t, Max_list, L, nu, axs = [], method = ''):
    N_data = len(Max_list)
    
    if len(axs) == 0:
        fig, axs = plt.subplots(2, 1,  figsize = (6, 9)) 
        axs[0].set_xlabel('t', fontsize=15)
        axs[0].set_ylabel(r'$\left|u_k\right|^2$', fontsize=15)
        axs[1].set_xlabel('k', fontsize=15)
        axs[1].set_ylabel(r'$\tau_d k^2$', fontsize=15)
        for ax in axs:
            ax.grid(alpha = 0.3)
            ax.minorticks_on()
            ax.tick_params('x', which='major', direction='in', length=5)
            ax.tick_params('y', which='major', direction='in', length=5)
            ax.tick_params('y', which='minor', direction='in', length=3, left=True)
        new_axs = True
    else: new_axs = False
    ax = axs[0]
    list_tau = []
    list_tau_teo = []
    list_dtau = []
    list_k = []
    for i in range(0, N_data):
        m_mode = i + 1
        data = Max_list[:, i]
        m = data > 1e-7
        data = data[m]
        aux_t = np.copy(t[m])
        if len(data) < 50:
            break
        print(len(Max_list))
        def exp_func(t, tau, A):
            return A*np.exp(-2*t/tau)
        init = [10, 1] 
    
        xx = np.linspace(np.min(aux_t), np.max(aux_t), 1000)
#        plt.plot(xx, exp_func(xx, *init))
#        plt.scatter(t, Max_list)
#        plt.show()
        pars, covm = curve_fit(exp_func, aux_t, data, init, absolute_sigma=False)
    #    chi2 =((w*(chi-fit_func(L,*pars))**2)).sum()
    #    ndof = len(L) - len(init)
        errors = np.sqrt(np.diag(covm))
    
        tau, dtau = pars[0], errors[0]
        A, dA = pars[1], errors[1]
#        B, dB = pars[2], errors[2]
        k = 2*np.pi/L * (i+1)
        tau_teo = 1/(k**2 * nu)
        print(tau_teo, tau)
        if i < 8 and new_axs:
            ax.set_title(r'Analisi delle densità spettrali di energia $\left|u_k\right|^2$ nel tempo')
            ax.scatter(aux_t[::200], data[::200], label = f'm = {m_mode}', s = 10)
            ax.plot(xx, exp_func(xx, tau, A), lw = 0.5)
        list_tau.append(tau)
        list_tau_teo.append(tau_teo)
        list_dtau.append(dtau)
        list_k.append(k)
#    axs[0].set_yscale('log')
#    axs[0].legend()
    ax = axs[1]
    list_k = np.array(list_k)
    list_tau = np.array(list_tau)
    list_tau_teo = np.array(list_tau_teo)
    list_dtau = np.array(list_dtau)
    def fit_tau(k, A, alpha):
        return A*k**(-alpha) 
    init = [1, 1]
    sigma = list_dtau
    w = 1/sigma**2
    pars, covm = curve_fit(fit_tau, list_k, list_tau, init, w, absolute_sigma=False)
    errors = np.sqrt(np.diag(covm))
    A, dA = pars[0], errors[0]
    alpha, dalpha = pars[1], errors[1]
    if new_axs:
        ax.scatter(list_k, list_tau_teo*list_k**2, s = 7, label = r'$\tau_d k^2$ teorico')
        ax.plot(list_k, np.ones(len(list_k))* 1/nu, lw = 0.8, label = '1/nu')
    ax.scatter(list_k, list_tau*list_k**2, s = 10,  label = r'$\tau_d k^2$ da fit (' + method + ') ')
    ax.set_ylim(200, 300)
#    axs[1].legend()
#    ax.plot(list_k, np.ones(len(list_k))*(1/nu))
#    plt.show()
    return axs, alpha, dalpha




if __name__ == '__main__':
    # Parameters of simulation
    L = 10           # Spatial size of the grid
    N = 200          # spatial step of grid
    dt = 5e-2        # Temporal step
    N_step = 10600     # Number of Temporal steps
    intermediate_step = 10000

    dx = L/N
    c = 0.1 
#    nu = 0.0000#31 # diffusion parameter
    nu = 4e-3

    # Numerical Methods
    I = Int.RKN
    RKorder = 4

    # initial condition
    init_function = gaussian
    m = 3
    init_params = [L/2, 0.2]
    l1 = [der.fft_der]
    l2 = [der.fft_der2]
    name_der = ['fft']
    t = np.linspace(0, dt*N_step, intermediate_step)
    M_array = check_derivative_effect(L, N, nu, dt, N_step, c,
                                      init_function, advection_diffusion, l1, l2, name_der, I, 
                                      RKorder, init_params, intermediate_step=intermediate_step,
                                      return_max=True)
    axs, alpha_fft, dalpha_fft = fit_Max_gaussian(t, M_array, L, nu, method=name_der[0])
    l1 = [der.simm_der]
    l2 = [der.simm_der2]
    name_der = ['dfs']
    M_array = check_derivative_effect(L, N, nu, dt, N_step, c,
                                      init_function, advection_diffusion, l1, l2, name_der, I, 
                                      RKorder, init_params, intermediate_step=intermediate_step,
                                      return_max=True)
    axs, alpha_dfs, dalpha_dfs = fit_Max_gaussian(t, M_array, L, nu, axs = axs, method=name_der[0])
    axs[0].legend(fontsize = 12)
    axs[1].legend(loc = 'lower left', fontsize = 12)
    plt.savefig('../figures/final/fit_tau_d.png', dpi = 200)
    plt.show()
    print('dfs')
    print(alpha_dfs, dalpha_dfs) 
    print('dfs')
    print(alpha_fft, dalpha_dfs)
