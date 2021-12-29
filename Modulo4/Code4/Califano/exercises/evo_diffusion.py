"""
This file uses some integrators to solve the equation:
    d_t ( u ) = nu * d_x^2 ( u ) - c * d_x ( F ( u ) )
Where d_t and d_x are the derivative respect t and x.
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import NumDerivative as der
import NumIntegrator as Int

# Define the right hand side term of the eq
def F_diffusion(u, c, nu, dx):
    """
    Function of equation:
         du/dt = nu * d^2u/d^2x - c * du/dx = F(u)
    """
    return nu * der.simm_der2(u, dx) - c * der.simm_der(u, dx)

# Define the initial condition
def u_sin_simple(x, L):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    k = 1
    return np.sin(k * 2*np.pi*x/L)
    
def u_sin_hard(x, L):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    k1 = 1
    k2 = 10
    k3 = 20
    return np.sin(k1 * 2*np.pi*x/L) + 1/5*np.sin(k2 * 2*np.pi*x/L) + 1/10*np.sin(k3 * 2*np.pi*x/L)

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
    N = 100 # spatial step of grid
    nu = 0.05 # useless parameter
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 25 # Number of Temporal steps
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)
    
    speed = dx/dt # speed given the values
    c = speed
    
    # Define starting function
    u_init = u(x, L) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
    # Fourier transform in time 
    c = c/2
    von_neumann = nu * dt /(dx**2)
    print(f'Von Neumann factor {von_neumann}')
    def my_real_fft(u, dx):
        """
        Real Fast Fourier Transform.
        """
        # x : 2 * pi = y : 1 ---> unitary rate
        # => dy = dx/(2*pi)
        dy =  dx / (2 * np.pi)
        # fft(j) = (u * exp(-2*pi*i*j*np.arange(n)/n)).sum()
        fft = fftpack.fft(u) # Discret fourier transform 
        k = fftpack.fftfreq(N, dy) 
        return fft, k
    
    # Amplitude**2 of Fourier coefficients #
    for i in range(N_step):# Evolution in time
        u_t = int_method(u_t, F, dt, c, nu, dx)
    
    fft_u_init, k_u_init = my_real_fft(u_init, dx)# FFT of initial u 
    fft_u, k_u = my_real_fft(u_t.real, dx) # FFT of final u 
 
    mod_fft2_u = (np.abs(fft_u)/N)**2 # square of module of fft final
    mod_fft2_u_init = (np.abs(fft_u_init)/N)**2 # square of module of fft final

    mask_pos_k_u = k_u >= 0 # mask for positive k final
    mod_fft2_u = mod_fft2_u[mask_pos_k_u]
    k_u = k_u[mask_pos_k_u]

    mask_pos_k_u_init = k_u_init >= 0 # mask for positive k init
    mod_fft2_u_init = mod_fft2_u_init[mask_pos_k_u_init]
    k_u_init = k_u_init[mask_pos_k_u_init]

   
    # Plot results
    plt.figure(figsize=(10, 6))
    equation = r'$\partial_t u = \nu \partial_x^2 u - c \partial_x u \ \longrightarrow$  Derivative with simm. finite difference'
    integ_params = '\n\nIntegrated with Runge Kutta of order 2'
    sistem_params = f'\n\nL = {L},    dx = {dx},    dt = {dt},    N step = {N_step},    c = {c},    nu = {nu}'
    stability = f'\n\nVon Neumann Factor = {von_neumann:.2f},    CFL (c dt/dx) = {c*dt/dx}'
    plt.suptitle(equation + integ_params + sistem_params + stability, fontsize = 15)
    plt.subplot(121)
    plt.plot(x, u_init, label = 'init')
    plt.plot(x, u_t.real, label = 'final')
    plt.xlabel('x', fontsize=15)
    plt.ylabel('u', fontsize=15)
    plt.legend(fontsize=13)
    plt.subplot(122)
    plt.scatter(k_u_init, mod_fft2_u_init, label = 'init')
    plt.scatter(k_u, mod_fft2_u, label = 'final')
    plt.xlabel('k', fontsize=15)
    plt.ylabel(r'$\left|u_k\right|^2$', fontsize=15)
    plt.legend(fontsize=13)
    plt.yscale('log')
    plt.tight_layout()
#    plt.savefig('figures/6_evo_diffusion_hard_function.png', dpi = 200)
    plt.show()
    
if __name__ == '__main__':
    ampl2(Int.RK2, u_sin_hard, F_diffusion)
