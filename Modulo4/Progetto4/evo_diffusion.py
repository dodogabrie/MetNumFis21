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
from utils import plot_template, my_real_fft, evaluate_energy_density_spectrum

# Define the right hand side term of the eq
def F_diffusion(u, c, nu, dx):
    """
    Function of equation:
         du/dt = nu * d^2u/d^2x - c * du/dx = F(u)
    """
    return nu * der.fft_der2(u, dx) #- c * der.simm_der(u, dx)

# Define the initial condition
def u_sin_simple(x,L, k = 1):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    return np.sin(k * 2*np.pi*x/L)
    
def u_sin_hard(x, L, k = 1):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    k1 = 1*k
    k2 = 10*k
    k3 = 20*k
    return np.sin(k1 * 2*np.pi*x/L) + 1/5*np.sin(k2 * 2*np.pi*x/L) + 1/10*np.sin(k3 * 2*np.pi*x/L)

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
    nu = 0.0003 # useless parameter
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 7500 # Number of Temporal steps
    
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

    # Amplitude**2 of Fourier coefficients #
    for i in range(N_step):# Evolution in time
        u_t = int_method(u_t, F, dt, c, nu, dx)

    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx) 
    mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t, dx)
   
    # Plot results
    fig, axs = plot_template()
    equation = r'$\partial_t u = \nu \partial_x^2 u - c \partial_x u \ \longrightarrow$  RK2 e differenze finite simmetriche'
    plt.suptitle(equation, fontsize = 15)

    ax = axs[0]
    ax.plot(x, u_init, label = 'init')
    ax.plot(x, u_t.real, label = 'final')
    ax = axs[1]
    ax.scatter(k_u_init, mod_fft2_u_init, label = 'init')
    ax.scatter(k_u, mod_fft2_u, label = 'final')
    plt.tight_layout()
#    plt.savefig('figures/6_evo_diffusion_hard_function.png', dpi = 200)
    plt.show()


def nu_effect(int_method, u, F):
    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 5000 # Number of Temporal steps
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)
    
    speed = dx/dt # speed given the values
    c = speed
    
    # Define starting function
    m = 1
    u_init = u(x, L, k = m) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
    # Fourier transform in time 
    c = c/2

    fig, axs = plot_template(2, 1, figsize = (5, 7))
    # |u_k|^2
    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx) 
    
    # generate initial condition with all the m
    list_m = (k_u_init/(2*np.pi)*L).astype(int)
    u_init_all_k = 0
    for m in list_m:
        u_init_all_k += 1/(m)**2 * u(x, L, k = m)
    u_init = np.copy(u_init_all_k)

    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx) 

    ax = axs[0]
    ax.plot(x, u_init, label = 'init')

    ax = axs[1]
    ax.set_ylabel(r'$G(k, t)$')
    ax.scatter(k_u_init, mod_fft2_u_init/mod_fft2_u_init, label = 'init', s = 10)
#    list_nu = np.logspace(-2, -3, 2)
    list_nu = np.array([1e-2, 1e-3, 3e-4, 1e-4])

    r_c = c * dt / ( 2 * dx )
    
    u_k = np.sqrt(mod_fft2_u_init[-1]) # |u_k|
    k_m = k_u_init[-1]

    t_diff = lambda nu: 1/(nu * u_k * k_m**2)

    print(f'r_c: {r_c}')
    print(f'final physic time: {N_step*dt}')

    for nu in list_nu:
        u_t = np.copy(u_init)
        von_neumann = nu * dt /(dx**2)
        print(f'Von Neumann factor {von_neumann:.1e}')

        r_nu = nu * dt / ( dx * dx )
        print(f'nu: {nu:.1e};   r_nu: {r_nu:.2e}')
        print(f't diff.: {t_diff(nu):.2e}')
        ax = axs[0]
        for i in range(N_step):# Evolution in time
            u_t = int_method(u_t, F, dt, c, nu, dx)
    
        ax.plot(x, u_t.real, label = f'nu = {nu}')
        ax = axs[1]
        mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t, dx)
        ax.scatter(k_u, mod_fft2_u/mod_fft2_u_init, label = f'nu = {nu}', s = 10)

    # Plot results
    plt.suptitle(r'$\partial_t u = \nu\partial^2_x u$ $\rightarrow$ Derivata seconda con FFT + RK2')
    plt.legend(fontsize = 9)
    plt.tight_layout()
#    plt.savefig('figures/diffusion/diffusion_varying_nu', dpi = 200)
    plt.show()


# RHS with derivative as argument
def F_diffusion_der(u, c, nu, dx, der):
    """
    Function of equation:
         du/dt = nu * d^2u/d^2x - c * du/dx = F(u)
    """
    return nu * der(u, dx) #- c * der.simm_der(u, dx)

def analytic_sol(ampl_list, k_list, x, nu, T):
    sol = 0
    for A, k in zip(ampl_list, k_list):
        sol += A * np.exp( -nu * k**2 * T) * np.sin(k*x)
    return sol

def derivative_effect(int_method, u, F):
    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 5000 # Number of Temporal steps
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)
    
    speed = dx/dt # speed given the values
    c = speed
    
    # Define starting function
    m = 1
    u_init = u(x, L, k = m) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
    # Fourier transform in time 
    c = c/2

    fig, axs = plt.subplots(3,1, figsize = (5,8))
    for ax in axs:
       ax.grid(alpha = 0.3)
       ax.minorticks_on()
       ax.tick_params('x', which='major', direction='in', length=5)
       ax.tick_params('y', which='major', direction='in', length=5)
       ax.tick_params('y', which='minor', direction='in', length=3, left=True)
       ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
       ax.set_xlabel('k', fontsize=15)
       ax.set_ylabel(r'$G(k, t)$')
       ax.set_yscale('log')
       ax.set_xscale('log')
 
    # |u_k|^2
    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx) 
    
    # generate initial condition with all the m
    list_m = (k_u_init/(2*np.pi)*L).astype(int)
    u_init_all_k = 0
    for m in list_m:
        u_init_all_k += 1/(m)**2 * u(x, L, k = m)
    u_init = np.copy(u_init_all_k)

    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx) 

    for ax in axs:
        ax.scatter(k_u_init, mod_fft2_u_init/mod_fft2_u_init, label = 'init', s = 10)
        ax.plot(k_u_init, mod_fft2_u_init/mod_fft2_u_init, lw = 0)

    list_nu = np.array([1e-2, 1e-3, 3e-4, 1e-4])

    r_c = c * dt / ( 2 * dx )
    
    u_k = np.sqrt(mod_fft2_u_init[-1]) # |u_k|
    k_m = k_u_init[-1]
    t_diff = lambda nu: 1/(nu * u_k * k_m**2)

    print(f'r_c: {r_c}')
    print(f'final physic time: {N_step*dt}')
    der_methods = [der.simm_der2, der.diff_fin_comp_der2, der.fft_der2]
    der_methods_names = ['Differenze finite simmetriche', 
                         'Differenze finite compatte', 
                         'Derivata con FFT']
    for ax, der_method, title in zip(axs, der_methods, der_methods_names):
        for nu in list_nu:
            u_t = np.copy(u_init)
            von_neumann = nu * dt /(dx**2)
            r_nu = nu * dt / ( dx * dx )
            for i in range(N_step):# Evolution in time
                u_t = int_method(u_t, F, dt, c, nu, dx, der_method)
        
            mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t, dx)
            ax.scatter(k_u, mod_fft2_u/mod_fft2_u_init, label = f'nu = {nu}', s = 6)

            ana_sol = analytic_sol([1/m**2 for m in list_m], k_u_init, x, nu, dt*N_step)
            mod_fft2_u, k_u = evaluate_energy_density_spectrum(ana_sol, dx)
            ax.plot(k_u, mod_fft2_u/mod_fft2_u_init, lw = 0.8)
            ax.set_title(title)

    # Plot results
    plt.suptitle(r'$\partial_t u = \nu\partial^2_x u$ $\rightarrow$ RK2')
    plt.legend(fontsize = 9, frameon=False)
    plt.tight_layout()
    plt.savefig('figures/diffusion/diffusion_varying_der', dpi = 200)
    plt.show()
    return

def tau_diff_study(int_method, u, F):
    #define input variable
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 30000 # Number of Temporal steps
    t = np.linspace(0, dt*N_step, N_step, endpoint=False)
    
    nu = 1e-2

    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)
    
    speed = dx/dt # speed given the values
    c = speed
    
    # Define starting function
    m = 1
    u_init = u(x, L, k = m) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time
    # Fourier transform in time 
    c = c/2

#    fig, axs = plot_template(2, 1, figsize = (5, 7))
    # |u_k|^2
    mod_fft2_u_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx) 
#    ax = axs[0]
#    ax.plot(x, u_init, label = 'init')

#    ax = axs[1]
#    ax.set_ylabel(r'$G(k, t)$')
#    ax.scatter(k_u_init, mod_fft2_u_init/mod_fft2_u_init, label = 'init', s = 10)
#    list_nu = np.logspace(-2, -3, 2)
#    list_nu = np.array([1e-2, 1e-3, 3e-4, 1e-4])

    
    ################# Diff time ###############################
    u_k = np.sqrt(mod_fft2_u_init[m-1]) # |u_k|
    k_m = k_u_init[m-1]
    def t_diff(nu, k_m):
        return 1/(nu * k_m**2)
    ###########################################################

#    print(f'r_c: {r_c}')
    print(f'final physic time: {N_step*dt}')

    von_neumann = nu * dt /(dx**2)
    print(f'Von Neumann factor {von_neumann:.1e}')

    fig, ax = plt.subplots(1,1, figsize = (6, 6))

    print(L * np.max(k_u_init)/ (2 * np.pi))
    list_m = [1, 3, 15]
    ax.plot([], [], ' ', label=fr"$\nu$ = {nu}")
    for m in list_m:
        u_init = u(x, L, k = m) # save initial condition 
        k_m = 2 * np.pi / L * m
        print(f'm : {m} --> t diff.: {t_diff(nu, k_m):.2e}')
        u_t = np.copy(u_init)
        list_ampl_diff = []
        for i in range(N_step):# Evolution in time
            u_t = int_method(u_t, F, dt, c, nu, dx)
            mod_fft2_u, k_u = evaluate_energy_density_spectrum(u_t, dx)
            list_ampl_diff.append(mod_fft2_u[m-1])

        list_ampl_diff = np.array(list_ampl_diff) 
        max_ampl = np.max(list_ampl_diff )
        ax.plot(t, list_ampl_diff, label = f'm: {m}')
        idx_t_close = np.argmin(np.abs(t - t_diff(nu, k_m)))
        ymax = list_ampl_diff[idx_t_close] * 3
        ymin = list_ampl_diff[idx_t_close] / 3
        ax.vlines(x=t_diff(nu, k_m), ymin = ymin, ymax = ymax, lw = 0.8, color = 'k') 

    ax.tick_params(axis='both', labelsize=11)
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.set_title(r'Evoluzione di $\left|u_k\right|^2$ nel tempo per diversi valori di $m$', fontsize = 13)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 10 * max_ampl)
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
    ax.set_xlabel('t', fontsize = 15)
    ax.set_ylabel(r'$\left|u_k\right|^2$', fontsize = 15)
    plt.legend( fontsize = 12)
#    plt.savefig('figures/diffusion/diff_time_varying_m.png', dpi = 200)
    plt.show()
 
 


if __name__ == '__main__':
    #ampl2(Int.RK2, u_sin_simple, F_diffusion)
    #nu_effect(Int.RK2, u_sin_simple, F_diffusion)
    #derivative_effect(Int.RK2, u_sin_simple, F_diffusion_der)
    tau_diff_study(Int.euler_step, u_sin_simple, F_diffusion)
