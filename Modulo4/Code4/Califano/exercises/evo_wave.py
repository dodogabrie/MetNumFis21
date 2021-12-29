"""
This file uses the Euler integrator to solve the equation:
    d_t ( u ) = - c * d_x ( F ( u ) )
Where d_t and d_x are the derivative respect t and x.
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import NumDerivative as der
import NumIntegrator as Int

# Define the right hand side term of the eq
def F(u, c, dx):
    """
    Function of equation:
         du/dt = - c * du/dx = F(u)
    """
    return - c * der.fft_der(u, dx)

# Define the initial condition
def u_sin_simple(x, L, A = 1.):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    k = 1
    return A * np.sin(k * 2*np.pi*x/L)

def u_sin_hard(x, L):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    k1 = 2
    k2 = 4
    k3 = 8
    return np.sin(k1 * 2*np.pi*x/L) + np.sin(k2 * 2*np.pi*x/L) + np.sin(k3 * 2*np.pi*x/L)

def evo_varying_speed(int_method, u):
    """
    This function vary the speed of the evolution (c) observing the stability
    of the algorithm: if c grows more than dx/dt the algorithm bevame unstable.

    Parameters
    ----------
    int_method: function 
        Integration method (euler or RK2).
    u: function
        Initial condition
    """
    def F(u, c, dx):
        """
        Function of equation:
             du/dt = c * du/dx = F(u)
        """
        return c * der.simm_der(u, dx)

    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    nu = 0.01 # useless parameter
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 20 # Number of Temporal steps
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)
    
    speed = dx/dt # speed given the values
    c = speed
    
    # Define starting function
    u_init = u(x, L) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
    
    ###############################################################
    # Evolution varying c, if c > dx/dt the evolution is unstable
    
    facs = np.linspace(0.1, 3, 3) # array of factors
    for fac in facs:
        c = speed*fac
        for i in range(N_step): # temporal evolution
            u_t = int_method(u_t, F, dt, c, dx)
        plt.plot(x, u_t, label = f'final c * {fac:.1f}')
    plt.plot(x, u_init, label = 'init')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()

###############################################################
def ampl2(int_method, u):
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
    """
    def F(u, c, dx):
        """
        Function of equation:
             du/dt = c * du/dx = F(u)
        """
        return - c * der.fft_der(u, dx)

    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    nu = 0.01 # useless parameter
    
    # temporal inputs
    dt = 0.05 # Temporal step
    N_step = 50 # Number of Temporal steps
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)
    
    speed = dx/dt # speed given the values
    c = speed
    
    # Define starting function
    u_init = u(x, L, A = 1) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
    # Fourier transform in time 
    c = c/2
    
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
        u_t = int_method(u_t, F, dt, c, dx)
    
    fft_u_init, k_u_init = my_real_fft(u_init, dx)# FFT of initial u 
    fft_u, k_u = my_real_fft(u_t, dx) # FFT of final u 

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
    equation = r'$\partial_t u = - c \partial_x u \ \longrightarrow$  Derivative with FFT'
    integ_params = '\n\nIntegrated with Runge Kutta of order 2'
    sistem_params = f'\n\nL = {L},    dx = {dx},    dt = {dt},    N step = {N_step},    dx/dt = {dx/dt},    c = {c}'
    plt.suptitle(equation + integ_params + sistem_params, fontsize = 15)
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
    #plt.savefig('figures/1_evo_wave_fft.png', dpi = 200)
    plt.show()

if __name__ == '__main__':
    #evo_varying_speed
    #ampl2
    ampl2(Int.RK2, u_sin_simple)
