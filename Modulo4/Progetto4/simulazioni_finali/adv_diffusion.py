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

def evolve_advection_diffusion(L, N, nu, dt, N_step, c,
                               u, F, der1, der2, int_method, RKorder, 
                               init_params):
    dx, x, u_init, u_t = initialization(L, N, u, init_params)
    for _ in range(N_step): # temporal evolution
            u_t = int_method(u_t, F, dt, RKorder, c, nu, dx, der1, der2) 
            u_k2, _ = evaluate_energy_density_spectrum(u_t, dx)
    return u_init, u_t, x, dx


