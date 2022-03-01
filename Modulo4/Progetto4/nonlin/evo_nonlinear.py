"""
This file uses some integrators to solve the non-linear equation:
    d_t ( u ) = - u * d_x ( F ( u ) )
Where d_t and d_x are the derivative respect t and x.
"""
import sys
sys.path.append('../')

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
def u_sin_simple(x, L, k = 1):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    return np.sin(k * 2*np.pi*x/L)
    
def u_sin_hard(x, L):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    k1 = 1
    k2 = 10
    k3 = 20
    return np.sin(k1 * 2*np.pi*x/L) + 1/5*np.sin(k2 * 2*np.pi*x/L) + 1/10*np.sin(k3 * 2*np.pi*x/L)

# Fourier transform in time 
def my_real_fft(u, dx):
    """
    Real Fast Fourier Transform.
    """
    N = len(u)
    # x : 2 * pi = y : 1 ---> unitary rate
    # => dy = dx/(2*pi)
    dy =  dx / (2 * np.pi)
    # fft(j) = (u * exp(-2*pi*i*j*np.arange(n)/n)).sum()
    fft = fftpack.fft(u) # Discret fourier transform 
    k = fftpack.fftfreq(N, dy) 
    return fft, k

def initial(u, m, kind = 'classic'):
    """
    kind: 
        - classic : classical boundary condition
        - extra_pt: add two point to the grid :)
    """
    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    nu = 0.05 # useless parameter
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 15 # Number of Temporal steps
    t = np.linspace(0, dt * N_step, N_step+1)

    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    if kind == 'classic':
        x = np.linspace(0,L,N, endpoint = False)
    elif kind == 'extra_pt':
        x = np.linspace(0, L + 2*dx, N+2, endpoint = False)
    else: 
        raise Exception("kind {kind} not found, pass 'classic' or 'extra_pt")

    # Define starting function
    u_init = u(x, L, k = m) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
    return L, N, nu, dt, N_step, dx, x, t, u_init, u_t

def evaluate_energy_density_spectrum(u_t, dx):
    # final fft
    fft_u, k_u = my_real_fft(u_t, dx) # FFT of final u 
    mod_u_k = np.abs(fft_u) # |u_k|
    mod_fft2_u = (mod_u_k)**2 # square of module of fft final
    mask_pos_k_u = k_u > 0 # mask for positive k final
    mod_fft2_u = mod_fft2_u[mask_pos_k_u]
    k_u = k_u[mask_pos_k_u]
    return mod_fft2_u, k_u

def plot_template():
    fig, axs = plt.subplots(1, 2, figsize = (12, 3))
    ax = axs[0]
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('u', fontsize=15)
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
    ax = axs[1]
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
    ax.set_xlabel('k', fontsize=15)
    ax.set_ylabel(r'$\left|u_k\right|^2$', fontsize=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return fig, axs

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
    L, N, nu, dt, N_step, dx, x, t, u_init, u_t = initial(u, m = 1)

    # initial fft
    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx)
    
    fig, axs = plot_template()
    ## First line
    for i in range(N_step):# Evolution in time
        u_t = int_method(u_t, F, dt, dx)
    ax = axs[0]
    ax.plot(x, u_init, label = 't = 0')
    ax.plot(x, u_t.real, label = f't = {N_step * dt:.1f}')

    ax = axs[1]
    # final fft
    mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t, dx)
    ax.scatter(k_u_init, mod_fft2_u_init, label = 't = 0', marker='.', zorder = 2, s = 12)
    ax.scatter(k_u, mod_fft2_u, label = f't = {N_step * dt}', marker = 'v', zorder = 1, s = 15)

    # Second line
    for i in range(N_step):# Evolution in time
        u_t = int_method(u_t, F, dt, dx)
    ax = axs[0]
    ax.plot(x, u_t.real, label = f't = {2 * N_step * dt:.1f}')

    ax = axs[1]
    # final fft
    mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t, dx)
    ax.scatter(k_u, mod_fft2_u, label = f't = {2 * N_step * dt}', marker = 'v', zorder = 1, s = 15)

        
    # Plot results
    equation = r'$\partial_t u = - u \partial_x u \ \longrightarrow$  differenze finite simmetriche + RK2'
    plt.suptitle(equation, fontsize = 15)
    [ax.legend(fontsize = 12) for ax in axs]
    plt.tight_layout()
#    plt.savefig('../figures/nonlinear/simm_RK2.png', dpi = 200)
    plt.show()



def RHS(t, x):
    return 0 * x

def F_flux(u):
    """
    Function of equation:
         du/dt = -dF/dx => F(u) = u*u/2
    """
    return u**2/2
 

def test_Lax_Wendroff(u):

    k = 1
    L, N, nu, dt, N_step, dx, x, t, u_init, u_t = initial(u, m = k, kind = 'extra_pt')

    # Line 1
    fig, axs = plot_template()
    for i in range(N_step): # temporal evolution
        u_t = NS.Lax_W_Two_Step(u_t, x, t, dt, dx, F_flux, RHS)
    ax = axs[0]
    ax.plot(x[:-2], u_init[:-2], label = f't=0')
    ax.plot(x[:-2], u_t[:-2], label = f't={t[-1]:.1f}')

    ax = axs[1]
    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init[:-2], dx)
    mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t[:-2], dx)
    ax.scatter(k_u_init, mod_fft2_u_init, label = 't = 0', marker='.', zorder = 2, s = 12)
    ax.scatter(k_u, mod_fft2_u, label = f't = {t[-1]:.1f}', marker = 'v', zorder = 1, s = 15)

    
    # Line 2
    for i in range(N_step): # temporal evolution
        u_t = NS.Lax_W_Two_Step(u_t, x, t, dt, dx, F_flux, RHS)
    for i in range(N_step): # temporal evolution
        u_t = NS.Lax_W_Two_Step(u_t, x, t, dt, dx, F_flux, RHS)

    ax = axs[0]
    ax.plot(x[:-2], u_t[:-2], label = f't={3 * t[-1]:.1f}')

    ax = axs[1]
    mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t[:-2], dx)
    ax.scatter(k_u, mod_fft2_u, label = f't = {3 * t[-1]:.1f}', marker = 'v', zorder = 1, s = 15)

    equation = r'$\partial_t u = - u \partial_x u \ \longrightarrow$  Lax-Wendroff'
    plt.suptitle(equation, fontsize = 15)
    [ax.legend(fontsize = 12) for ax in axs]
    plt.tight_layout()
#    plt.savefig('../figures/nonlinear/Lax_Wendroff.png', dpi = 200)
    plt.show()

def test_turn_over(u):

    k = 1
    L, N, nu, dt, N_step, dx, x, t, u_init, u_t = initial(u, m = k, kind = 'extra_pt')

    u_init = u(x, L, k = k) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time

    # initial fft
    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init[:-2], dx)
    
    # Turnover time
    mod_u_k_init = np.sqrt(mod_fft2_u_init)
    u_k_in = mod_u_k_init[k-1]
    k_in = k_u_init[k-1]
    t_o = 1/(u_k_in*k_in)
    print(t_o)

    fig, axs = plot_template()

    ax = axs[1]
    N_inside = 10
    for under_turnover, over_turnover, label_t in [[20, 1, r'$\tau_o/20$'], 
                                                   [1, 1, r'$\tau_o$'], 
                                                   [1, 20, r'$20\tau_o$']]:
        dt = t_o / (N_inside * under_turnover)
        N_step = N_inside * over_turnover * under_turnover
        # in this way dt * N_step = t_o
        for i in range(N_step): # temporal evolution
            u_t = NS.Lax_W_Two_Step(u_t, x, t, dt, dx, F_flux, RHS)
        mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t[:-2], dx)
        ax.scatter(k_u, mod_fft2_u, label = fr't = ' + label_t, marker = 'v', zorder = 1, s = 12)

    # Plot results
    equation = r'$\partial_t u = - u \partial_x u \ \longrightarrow$ Lax-Wendroff'
    plt.suptitle(equation, fontsize = 15)

    ax = axs[0]
    ax.plot(x, u_init, label = 't = 0', c = 'purple')
    ax.plot(x, u_t.real, label = fr't = {over_turnover}$\tau_o$', c = 'lime')
    ax = axs[1]
    ax.scatter(k_u_init, mod_fft2_u_init, label = 't = 0', marker='D', zorder = 2, s = 12, c = 'purple')
    [ax.legend(fontsize = 12) for ax in axs]
    plt.tight_layout()
    #plt.savefig('../figures/nonlinear/tunnovetime.png', dpi = 200)
    plt.show()

if __name__ == '__main__':
    #test_Lax_Wendroff(u_sin_simple)
    ampl2(Int.RK2, u_sin_simple, F_non_linear)
    #test_turn_over(u_sin_simple)
