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

    check_CFL(c, dx, dt) # check CFL condition
    check_VonNeumann(nu, dt, dx) # Check Von Neumann on diffusion only

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

def fit_Max_gaussian(t, Max_list, L, nu):
    N_data = len(Max_list)
    for i in range(1, N_data):
        data = Max_list[:, i]
        m = data > 1e-7
        data = data[m]
        aux_t = np.copy(t[m])
        if len(data) < 10:
            break
        print(len(Max_list))
        def exp_func(t, tau, A, B):
            return A*np.exp(-2*t/tau) + B
        init = [10, 1, 1] 
    
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
        B, dB = pars[2], errors[2]
        k = 2*np.pi/L * (i+1)
        tau_teo = 1/(k**2 * nu)
        print(tau_teo, tau)
    
        plt.scatter(aux_t, data)
        plt.plot(xx, exp_func(xx, tau, A, B))
#        plt.yscale('log')
    plt.show()




if __name__ == '__main__':
    # Parameters of simulation
    L = 10           # Spatial size of the grid
    N = 200          # spatial step of grid
    dt = 5e-2        # Temporal step
    N_step = 10600     # Number of Temporal steps
    intermediate_step = 50

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
    M_array = check_derivative_effect(L, N, nu, dt, N_step, c,
                                      init_function, advection_diffusion, l1, l2, name_der, I, 
                                      RKorder, init_params, intermediate_step=intermediate_step,
                                      return_max=True)
    t = np.linspace(0, dt*N_step, intermediate_step)
    fit_Max_gaussian(t, M_array, L, nu)
