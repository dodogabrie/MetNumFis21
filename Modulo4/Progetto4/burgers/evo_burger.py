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
    
def ampl2(int_method, u, F):
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
    nu = 0.0025 # useless parameter
    
    # temporal inputs
    dt = 0.0005 # Temporal step
    N_step = 5550 # Number of Temporal steps
    der_method = 'fft'
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)

    # Define starting function
#    u_init = u(x, L) # save initial condition 
    m = 1
    u_init = u(x, L, k = m) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time

    von_neumann = nu * dt /(dx**2)
    print(von_neumann)

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

    for i in range(N_step):# Evolution in time
        u_t = int_method(u_t, F, dt, nu, dx, 
                         d1,  # first derivative
                         d2  # second derivative
                         )
    
    u_k2_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx)# FFT of initial u 
    u_k2, k_u = evaluate_energy_density_spectrum(u_t.real, dx) # FFT of final u 
    
    fig, axs = plot_template()
    equation = r'$\partial_t u = \nu \partial_x^2 u - u \partial_x u \ \longrightarrow$  Derivative with simm. finite difference'
    ax = axs[0]
    ax.plot(x, u_init, label = 'init')
    ax.plot(x, u_t, label = 'final')
    ax.scatter(x, u_init, s = 3)
    ax.scatter(x, u_t.real, s = 3)

    ax = axs[1]
    ax.scatter(k_u_init, u_k2_init, label = 'init')
    ax.scatter(k_u, u_k2, label = 'final')
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    ampl2(Int.RK2, u_sin_simple, F_burger)
