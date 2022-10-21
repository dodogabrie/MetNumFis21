"""
This file uses some integrators to solve the burger equation:
    d_t ( u ) = nu * d_x^2 ( u ) - u * d_x ( F ( u ) )
Where d_t and d_x are the derivative respect t and x.
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import NumDerivative as der
import NumIntegrator as Int

# Define the right hand side term of the eq
def F_burger(u, nu, dx):
    """
    Function of equation:
         du/dt = - u * du/dx = F(u)
    """
    return nu * der.simm_der2(u, dx) - u * der.simm_der(u, dx)

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
    N = 1000 # spatial step of grid
    nu = 0.005 # useless parameter
    
    # temporal inputs
    dt = 0.01 # Temporal step
    N_step = 300 # Number of Temporal steps
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)

    # Define starting function
    u_init = u(x, L) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time

    von_neumann = nu * dt /(dx**2)
    print(von_neumann)
    
    # Fourier transform in time 
    def my_real_fft(u, dx):
        """
        Real Fast Fourier Transform.
        """
        # x : 2 * pi = y : 1 ---> unitary rate
        # => dy = dx/(2*pi)
        dy =  dx / (2 * np.pi)
        # fft(j) = (u * exp(-2*pi*i*j*np.arange(n)/n)).sum()
        fft = fftpack.rfft(u) # Discret fourier transform 
        k = fftpack.rfftfreq(N, dy) 
        return fft, k
    
    for i in range(N_step):# Evolution in time
        u_t = int_method(u_t, F, dt, nu, dx)
    
    fft_u_init, k_u_init = my_real_fft(u_init, dx)# FFT of initial u 
    fft_u, k_u = my_real_fft(u_t.real, dx) # FFT of final u 
    
    # Plot results
    plt.figure(figsize=(10, 6))
    equation = r'$\partial_t u = \nu \partial_x^2 u - u \partial_x u \ \longrightarrow$  Derivative with simm. finite difference'
    integ_params = '\n\nIntegrated with Runge Kutta of order 2'
    sistem_params = f'\n\nL = {L},    dx = {dx},    dt = {dt},    N step = {N_step},    nu = {nu}'
    stability = f'\n\nVon Neumann Factor = nu dt/(dx*dx) = {von_neumann:.3f}'
    plt.suptitle(equation + integ_params + sistem_params + stability, fontsize = 15)
    plt.subplot(111)
    plt.plot(x, u_init, label = 'init')
    plt.plot(x, u_t.real, label = 'final')
    plt.xlabel('x', fontsize=15)
    plt.ylabel('u', fontsize=15)
    plt.legend(fontsize=13)
    
#    plt.subplot(122)
#    plt.plot(k_u_init, fft_u_init**2, label = 'init', marker='.',markersize = 10, lw = 0.)
#    plt.plot(k_u, fft_u**2, label = 'final', marker = '.', markersize = 10, lw = 0.)
#    plt.xlabel('k', fontsize=15)
#    plt.ylabel(r'$\left|u_k\right|^2$', fontsize=15)
#    plt.legend(fontsize=13)
#    plt.yscale('log')
#    plt.xscale('log')
    
#    plt.subplot(122)
#    plt.plot(k_u, fft_u**2 * k_u**2, label = r'$fft \cdot k^2$', marker = '.', markersize = 10, lw = 0., color = 'C1')
#    plt.xlabel('k', fontsize=15)
#    plt.ylabel(r'$\left|u_k\right|^2 \cdot k^2$', fontsize=15)
#    plt.legend(fontsize=13)
#    plt.yscale('log')
#    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('figures/8_evo_shocking_Burger.png', dpi = 200)
    plt.show()
    
if __name__ == '__main__':
    ampl2(Int.RK2, u_sin_simple, F_burger)
