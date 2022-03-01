"""
This file uses some integrators to solve the equation:
    d_t ( u ) = nu * d_x^2 ( u ) - c * d_x ( F ( u ) )
Where d_t and d_x are the derivative respect t and x.
"""
import sys
sys.path.append('../')

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import NumDerivative as der
import NumIntegrator as Int
from utils import plot_template, my_real_fft, evaluate_energy_density_spectrum

# Define the right hand side term of the eq
def F_diffusion(u, c, nu, dx):
    """
    Function of equation:
         du/dt = nu * d^2u/d^2x - c * du/dx = F(u)
    """
    return nu * der.simm_der2(u, dx) - c * der.simm_der(u, dx)

# Define the initial condition
def u_sin_simple(x,L, k = 1):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    return np.sin(k * 2*np.pi*x/L)

def ampl2(int_method, u, F):
    """
    This function plot the square of the Fourier coefficient of u 
    at the initial time and after few step.
    The bigger k are the first to explode.
    """
    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 5550 # Number of Temporal steps
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)
    
    speed = dx/dt # speed given the values

   
    # Plot results
    fig, axs = plot_template()
    equation = r'$\partial_t u = \nu \partial_x^2 u - c \partial_x u \ \longrightarrow$  Eulero e differenze finite simmetriche; $\tilde{c} = dx/dt$'
    plt.suptitle(equation, fontsize = 15)

    m = 1
    u_init = u(x, L, k = m) # save initial condition 
    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx) 
    ax = axs[0]
    ax.plot(x, u_init, label = 'init')
    ax = axs[1]
    ax.scatter(k_u_init, mod_fft2_u_init, label = 't = 0')

    c = speed
    list_c = [speed, speed/2, speed/100]
    list_symbol = ['v', 'p', 'x']
    for c, ms in zip(list_c, list_symbol): 
        # mandatory
        nu = c * dx /2 # Diffusion parameter in order to stabilize algorithm
        
        # Define starting function
        u_t = np.copy(u_init) # create e copy to evolve it in time
        # Fourier transform in time 
    
        von_neumann = nu * dt /(dx**2)
        print(f'Von Neumann factor {von_neumann}')
    
        # Amplitude**2 of Fourier coefficients #
        for i in range(N_step):# Evolution in time
            u_t = int_method(u_t, F, dt, c, nu, dx)
    
        mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t, dx)
        f = int(speed/c)
        if f == 1: str_name = r'$c =\tilde{c}$'
        else: str_name = r"c = $\tilde{c}$" + f'/{f}'
        ax = axs[0]
        ax.plot(x, u_t.real, label = str_name)
        ax = axs[1]
        ax.scatter(k_u, mod_fft2_u, label = str_name, marker = ms, s = 18)

    plt.tight_layout()
    plt.legend()
    plt.savefig('../figures/diff_adv/6_evo_diff_adv_varying_c.png', dpi = 200)
    plt.show()

if __name__ == '__main__':
    ampl2(Int.euler_step, u_sin_simple, F_diffusion)
