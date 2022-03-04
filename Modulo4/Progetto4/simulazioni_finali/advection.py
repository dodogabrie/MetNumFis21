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
                     init_params):
    dx, x, u_init, u_t = initialization(L, N, u, init_params)
    for _ in range(N_step): # temporal evolution
            u_t = int_method(u_t, F, dt, RKorder, c, nu, dx, der1, der2) 
            u_k2, _ = evaluate_energy_density_spectrum(u_t, dx)
    return u_init, u_t, x, dx

def check_derivative_effect(L, N, nu, dt, N_step, c,
                            u, F, list_der1, list_der2, name_der, int_method,
                            RKorder, init_params):
    CFL = c*dt/L * N
    title = f'L = {L},    N = {N},    dt = {dt:.1e},\n N_step = {N_step:.0e},    c = {c},    CFL = {CFL:.2f}'
    fig, axs = plot_template(2,1, figsize = (6, 9))
    plt.rc('font', **{'size'   : 15})
    plt.suptitle(title, fontsize = 15)
    for i, (der1, der2, name) in enumerate(zip(list_der1, list_der2, name_der)):
        results = evolve_advection(L, N, nu, dt, N_step, c,
                                   u, F, der1, der2, int_method, 
                                   RKorder, init_params)
        u_init, u_t, x, dx = results
        u_exact = analytic_sol_u_sin_simple_advection(x, m, c, N_step*dt)
        if i == 0: 
            plot_results(u_init, x, dx, label='init', axs = axs)
            plot_results(u_exact,x, dx, label='exact', axs = axs, ls = '--')
        axs = plot_results(u_t, x, dx, label = name, axs = axs)
    axs[0].legend(loc='lower right') 
    axs[1].legend(loc='center left') 
#    plt.savefig(f'../figures/final/advection_CFL{CFL:.2f}.png', dpi = 150)
#    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    # Parameters of simulation
    L = 10           # Spatial size of the grid
    N = 200          # spatial step of grid
    dt = 4.6e-1        # Temporal step
    N_step = 432     # Number of Temporal steps

    dx = L/N
    c = 0.1 
    nu = 0.0000#31 # diffusion parameter

    # Numerical Methods
    I = Int.RKN
    RKorder = 4

    # initial condition
    init_function = u_sin_simple 
    m = 3
    init_params = [L, m]
    l1 = [der.simm_der, der.fft_der]
    l2 = [der.simm_der2, der.fft_der2]
    name_der = ['dfs', 'fft']
    check_derivative_effect(L, N, nu, dt, N_step, c,
                            init_function, advection_diffusion, l1, l2, name_der, I, 
                            RKorder, init_params)
