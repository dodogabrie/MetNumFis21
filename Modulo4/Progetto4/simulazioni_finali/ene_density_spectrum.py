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

    # Define starting function
    u_init = u(x, init_params) # save initial condition 
    return dx, x, u_init


def print_uk(f, L, N, A, init_params):
    dx, _, u_init = initialization(L, N, f, init_params)
    u_init = A*u_init
    u_k2, _ = evaluate_energy_density_spectrum(u_init, dx)
    u_k = np.sqrt(u_k2)
    print(u_k[u_k>0.001])
    return 

if __name__ == '__main__':
    # Parameters of simulation
    L = 10           # Spatial size of the grid
    N = 200          # spatial step of grid
    
    m = 1
    A = 1
    init_function = u_sin_simple 
    init_params = [L, m]
    print_uk(init_function, L, N, A, init_params)


