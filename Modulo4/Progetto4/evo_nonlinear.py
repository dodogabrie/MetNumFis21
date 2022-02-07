"""
This file uses some integrators to solve the non-linear equation:
    d_t ( u ) = - u * d_x ( F ( u ) )
Where d_t and d_x are the derivative respect t and x.
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import NumDerivative as der
import NumIntegrator as Int
import NumScheme as NS

# Define the right hand side term of the eq
def F_non_linear(u, dx):
    """
    Function of equation:
         du/dt = - u * du/dx = F(u)
    """
    return - u * der.simm_der(u, dx)

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
    N_step = 40 # Number of Temporal steps
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)

    # Define starting function
    u_init = u(x, L) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time

    # Fourier transform in time 
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
        u_t = int_method(u_t, F, dt, dx)

    fft_u_init, k_u_init = my_real_fft(u_init, dx)# FFT of initial u 
    fft_u, k_u = my_real_fft(u_t, dx) # FFT of final u 

    mod_fft2_u = (np.abs(fft_u)/N)**2 # square of module of fft final
    mod_fft2_u_init = (np.abs(fft_u_init)/N)**2 # square of module of fft final

    mask_pos_k_u = k_u >= 0 # mask for positive k final
    mod_fft2_u = mod_fft2_u[mask_pos_k_u]
    k_u = k_u[mask_pos_k_u]

    # Plot results
    fig, axs = plt.subplots(1, 2, figsize = (15, 5))
    equation = r'$\partial_t u = - u \partial_x u \ \longrightarrow$  Derivative with simm. finite difference'
    integ_params = '\n\nIntegrated with Runge Kutta of order 2'
    sistem_params = f'\n\nL = {L},    dx = {dx},    dt = {dt},    N step = {N_step}'
    plt.suptitle(equation + integ_params + sistem_params, fontsize = 15)
    ax = axs[0]
    ax.plot(x, u_init, label = 'init')
    ax.plot(x, u_t.real, label = 'final')
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('u', fontsize=15)
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
    ax.legend(fontsize=13)

    ax = axs[1]
    ax.scatter(k_u_init, mod_fft2_u_init**2, label = 'init', marker='.', zorder = 2, s = 12)
    ax.scatter(k_u, mod_fft2_u**2, label = 'final', marker = 'v', zorder = 1, s = 15)
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
    ax.set_xlabel('k', fontsize=15)
    ax.set_ylabel(r'$\left|u_k\right|^2$', fontsize=15)
    ax.legend(fontsize=13)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.tight_layout()
    #plt.savefig('figures/7_evo_non_linear.png', dpi = 200)
    plt.show()

def test_Lax_Wendroff(u):
    def RHS(t, x):
        return 0 * x

    def F(u, c):
        """
        Function of equation:
             du/dt = -dF/dx => F(u) = u*u/2
        """
        return u**2/2

    # Fourier transform in time 
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
 
    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    nu = 0.01 # useless parameter
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 40 # Number of Temporal steps
    t = np.linspace(0, dt * N_step, N_step+1)
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0, L + 2*dx, N+2, endpoint = False)
    
    speed = dx/dt # speed given the values
    c = speed
    
    # Define starting function
    u_init = u(x, L) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
 
    for i in range(N_step): # temporal evolution
        u_t = NS.Lax_W_Two_Step(u_t, x, t, dt, dx, F, RHS, c)

    fft_u_init, k_u_init = my_real_fft(u_init[:-2], dx)# FFT of initial u 
    fft_u, k_u = my_real_fft(u_t[:-2], dx) # FFT of final u 

    mod_fft2_u = (np.abs(fft_u)/N)**2 # square of module of fft final
    mod_fft2_u_init = (np.abs(fft_u_init)/N)**2 # square of module of fft final

    mask_pos_k_u = k_u >= 0 # mask for positive k final
    mod_fft2_u = mod_fft2_u[mask_pos_k_u]
    k_u = k_u[mask_pos_k_u]

    fig, axs = plt.subplots(1, 2, figsize = (15, 5))
    ax = axs[0]
    ax.plot(x[:-2], u_init[:-2], label = f't=0')
    ax.plot(x[:-2], u_t[:-2], label = f'Lax Wendroff, t={t[-1]:.0f}')
    ax.grid(alpha = 0.3)
#    ax.set_xlim(0, L-dx)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.set_xlabel('x', fontsize = 15)
    ax.set_ylabel('u', fontsize = 15)
    suptitle ="   Soluzione dell'equazione di avvezione" 
    title = "Stabilità e condizione CLF"
    plt.suptitle(suptitle, fontsize = 15, y = 0.95)
    ax.set_title(title, fontsize = 13, y= 1. )
    ax.legend(fontsize = 12, loc = 'lower right')

    ax = axs[1]
    ax.scatter(k_u_init, mod_fft2_u_init**2, label = 'init', marker='.', zorder = 2, s = 12)
    ax.scatter(k_u, mod_fft2_u**2, label = 'final', marker = 'v', zorder = 1, s = 15)
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
    ax.set_xlabel('k', fontsize=15)
    ax.set_ylabel(r'$\left|u_k\right|^2$', fontsize=15)
    ax.legend(fontsize=13)
    ax.set_yscale('log')
    ax.set_xscale('log')
 
    plt.tight_layout()
#    plt.savefig('figures/advection/lw_advection.png', dpi = 200)
    plt.show()

if __name__ == '__main__':
    #test_Lax_Wendroff(u_sin_simple)
    ampl2(Int.RK2, u_sin_simple, F_non_linear)
