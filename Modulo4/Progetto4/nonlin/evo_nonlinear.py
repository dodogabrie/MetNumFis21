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
from utils import plot_template, evaluate_energy_density_spectrum, my_real_fft

# Define the right hand side term of the eq
def F_non_linear(u, dx):
    """
    Function of equation:
         du/dt = - u * du/dx = F(u)
    """
    return - u * der.fft_der(u, dx)

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


def initial(u, m, kind = 'classic'):
    """
    kind: 
        - classic : classical boundary condition
        - extra_pt: add two point to the grid :)
    """
    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 200 # spatial step of grid
    nu = 0.05 # useless parameter
    
    # temporal inputs
    dt = 0.01 # Temporal step
    N_step = 100 # Number of Temporal steps
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


def ampl2(int_method, order, u, F):
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

    title = f'L = {L},    N = {N},    dt = {dt:.1e}, N_step = {N_step:.0e}'
    fig, axs = plot_template(2,1, figsize = (6, 9))
    plt.rc('font', **{'size'   : 15})
    plt.suptitle(title, fontsize = 15)

    ## First line
    for i in range(N_step):# Evolution in time
        u_t = int_method(u_t, F, dt, order, dx)
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
        u_t = int_method(u_t, F, dt, order, dx)
    ax = axs[0]
    ax.plot(x, u_t.real, label = f't = {2 * N_step * dt:.1f}')

    ax = axs[1]
    # final fft
    mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t, dx)
    ax.scatter(k_u, mod_fft2_u, label = f't = {2 * N_step * dt}', marker = 'v', zorder = 1, s = 15)

        
    # Plot results
    equation = r'$\partial_t u = - u \partial_x u \ \longrightarrow$  differenze finite simmetriche + RK2'
    [ax.legend(fontsize = 12) for ax in axs]
#    plt.tight_layout()
    plt.savefig('../figures/final/nonlinear_fft.png', dpi = 200)
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
    ampl2(Int.RKN, 4, u_sin_simple, F_non_linear)
    #test_turn_over(u_sin_simple)
