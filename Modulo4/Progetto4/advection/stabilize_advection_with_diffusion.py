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

#    check_CFL(c, dx, dt) # check CFL condition
    check_VonNeumann(nu, dt, dx) # Check Von Neumann on diffusion only

    # Define starting function
    u_init = u(x, init_params) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
    return dx, x, u_init, u_t

def evolve_advection(L, N, nu, dt, N_step, c,
                     u, F, der1, der2, int_method, RKorder, 
                     init_params, check_amplitude = False):
    dx, x, u_init, u_t = initialization(L, N, u, init_params)
    list_uk2 = []
    last_uk2 = 1
    for _ in range(N_step): # temporal evolution
        if (last_uk2 > 1e-30) or (not check_amplitude):
            u_t = int_method(u_t, F, dt, RKorder, c, nu, dx, der1, der2) 

            u_k2, _ = evaluate_energy_density_spectrum(u_t, dx)
            list_uk2.append(max(1e-30, u_k2[init_params[1]-1]))
            last_uk2 = u_k2[init_params[1]-1]
        else: 
            list_uk2.append(1e-30)
    return u_init, u_t, list_uk2, x, dx

def check_derivative_effect(L, N, nu, dt, N_step, c,
                            u, F, list_der1, list_der2, int_method, 
                            RKorder, init_params):
    params_name = ['L = ', 'N = ', 'nu = ', 'dt = ', 'N_step = ', 'c = ']
    params = [L, N, nu, dt, N_step, c]
    title = ''
    for n, p in zip(params_name, params): title += n + f'{p:.0e};   '
    fig, axs = plot_template()
    plt.suptitle(title)
    for i, (der1, der2, name) in enumerate(zip(list_der1, list_der2, name_der)):
        results = evolve_advection(L, N, nu, dt, N_step, c,
                                   u, F, der1, der2, int_method, 
                                   RKorder, init_params)
        u_init, u_t, _, x, dx = results
        u_exact = analytic_sol_u_sin_simple_advection(x, m, c, N_step*dt)
        if i == 0: 
            plot_results(u_init, x, dx, label='init', axs = axs)
            plot_results(u_exact,x, dx,label='exact', axs = axs)
        axs = plot_results(u_t, x, dx, label = name, axs = axs)

    [ax.legend() for ax in axs]
#    plt.savefig('../figures/test_advection.png', dpi = 150)
    plt.show()
    return

def check_amplitude_uk(L, N, nu, dt, N_step, c,
                       u, F, list_der1, list_der2, int_method, 
                       RKorder, init_params):
    t = np.linspace(0, dt*N_step, N_step, endpoint=False)
    params_name = ['L = ', 'N = ', 'nu = ', 'dt = ', 'N_step = ', 'c = ']
    params = [L, N, nu, dt, N_step, c]
    title = ''
    for n, p in zip(params_name, params): title += n + f'{p:.0e};   '
    plt.figure()
    plt.suptitle(title)
    for i, (der1, der2, name) in enumerate(zip(list_der1, list_der2, name_der)):
        results = evolve_advection(L, N, nu, dt, N_step, c,
                                   u, F, der1, der2, int_method, 
                                   RKorder, init_params, check_amplitude = True)
        _, _, uk2, _, _ = results
        plt.plot(t, uk2, label = name)
        print(fit_exponential(t, uk2))
#    plt.savefig('../figures/test_advection.png', dpi = 150)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel(r'$\left|u_k\right|^2$')
    plt.legend()
    plt.show()
    return

def fit_exponential(t, u_k2):
    u_k2 = np.array(u_k2)
    mask_u = u_k2 > 1e-30
    tc = np.copy(t[mask_u])
    u_k2c = np.copy(u_k2[mask_u])
    def f(t, A, tau):
        return A*np.exp(-t/(2*tau))
    init = [1,1]
    pars, covm = curve_fit(f, tc, u_k2c, init)
    errors = np.sqrt(np.diag(covm))
#    A, dA = pars[0], errors[0]
    tau, dtau = pars[1], errors[1]
    return tau

def stabilize_fft_der(L, N, nu, dt, N_step, c,
                       u, F, int_method, 
                       RKorder, init_params):
    der1 = der.fft_der
    der2 = der.fft_der2
    results = evolve_advection(L, N, nu, dt, N_step, c,
                               u, F, der1, der2, int_method, RKorder, 
                               init_params, check_amplitude = False)
    u_init, u_t, _, x, dx = results
    fig, axs = plot_template()
    plot_results(u_init, x, dx, label='init', axs = axs)
    plot_results(u_t, x, dx, label=f't={N_step*dt}', axs = axs)
    plt.show()
    return 

def ana_meno_num(L, N, nu, dt, N_step, c,
                 u, F, der1, der2, int_method, 
                 RKorder, init_params):
    params = [L, N, nu, dt, N_step, c,
              u, F, der1, der2, int_method, RKorder, 
              init_params]
    m = init_params[1]
    fig, axs = plot_template()
    for i in range(0, 3):
        N_step = N_step + i * N_step
        params[4] = N_step
        results = evolve_advection(*params, check_amplitude = False)
        u_init, u_t, _, x, dx = results
#        if i == 1: plot_results(u_init, x, dx, label='init', axs = axs)
        u_ana = analytic_sol_u_sin_simple_advection(x, m, c, N_step*dt)
        plot_results(u_t-u_ana, x, dx, label=f't={N_step*dt}', axs = axs)
    [ax.legend() for ax in axs]
    plt.show()
    return

if __name__ == '__main__':
    # Parameters of simulation
    L = 20 # Spatial size of the grid
    N = 300 # spatial step of grid
    dt = 0.0013 # Temporal step
    N_step = 20000 # Number of Temporal steps

    dx = L/N
    c = dx/dt
    nu = 0.0 # diffusion parameter

    # Numerical Methods
    I = Int.RKN
    RKorder = 4

    # initial condition
    init_function = u_sin_simple 
    m = 6
    init_params = [L, m]

    test_der = 0
    ampl = 0
    stab_fft = 0
    ana_num = 0
    test_der_stability = 1
    if test_der_stability:
        c = 1
        name_der = ['dfs2', 'dfc']#, 'fft']
        list_der1 = [der.simm_der, der.diff_fin_comp_der]#, der.fft_der]
        list_der2 = [der.simm_der2, der.diff_fin_comp_der2]#, der.fft_der2]
        check_derivative_effect(L, N, nu, dt, N_step, c,
                                init_function, advection_diffusion, 
                                list_der1, list_der2, I, 
                                RKorder, init_params)
    if test_der:
        name_der = ['dfs2', 'dfs4', 'dfs6', 'dfc']
        list_der1 = [der.simm_der, der.simm_der_Ord4,
                     der.simm_der_Ord6, der.diff_fin_comp_der]
        list_der2 = [der.simm_der2, der.simm_der_Ord4_2, 
                     der.simm_der_Ord6_2, der.diff_fin_comp_der2]

        check_derivative_effect(L, N, nu, dt, N_step, c,
                                init_function, advection_diffusion, 
                                list_der1, list_der2, I, 
                                RKorder, init_params)
    if ampl:
        name_der = ['dfs2', 'dfs4', 'dfs6', 'dfc']
        list_der1 = [der.simm_der, der.simm_der_Ord4,
                     der.simm_der_Ord6, der.diff_fin_comp_der]
        list_der2 = [der.simm_der2, der.simm_der_Ord4_2, 
                     der.simm_der_Ord6_2, der.diff_fin_comp_der2]
        check_amplitude_uk(L, N, nu, dt, N_step, c,
                           init_function, advection_diffusion, 
                           list_der1, list_der2, I, 
                           RKorder, init_params)
    if stab_fft:
        stabilize_fft_der(L, N, nu, dt, N_step, c,
                           init_function, advection_diffusion, I, 
                           RKorder, init_params)
    if ana_num:
        der1 = der.simm_der
        der2 = der.simm_der2
        ana_meno_num(L, N, nu, dt, N_step, c,
                     init_function, advection_diffusion, 
                     der1, der2, I, 
                     RKorder, init_params)

