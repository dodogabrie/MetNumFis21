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
    
def ampl2(int_method, order, u, F, der_method = 'dfs', intermediate_step = 1):
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
    
    case1 = 1 # t_nl ~ t_d
    case2 = 0 # t_nl << t_d

    if case1:
        A = 1e-3
        nu = 0.1 # diffusive parameter
        dt = 0.0005
        N_step = 6000 # Number of Temporal steps
    elif case2:
        A = 1e-3
        nu = 0.000005 # diffusive parameter
        dt = 0.05 # Temporal step
        N_step = 9000 # Number of Temporal steps
    else:
        A = 1
        nu = 0.0055 # diffusive parameter
        dt = 0.005 
        N_step = 3*65500 # Number of Temporal steps

    print('physic time:', dt * N_step)
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)

    # Define starting function
#    u_init = u(x, L) # save initial condition 
    m = 5
    u_init = A*u(x, L, k = m) # save initial condition 
    u_t = np.copy(u_init) # create e copy to evolve it in time

#    von_neumann = nu * dt /(dx**2)

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
    
    u_k2_init, k_u_init = evaluate_energy_density_spectrum(u_init, dx)# FFT of initial u 
    u_k = u_k2_init[m-1]
    last_u_k = u_k2_init[-1]
    last_k = k_u_init[-1]
    
    k = 2*np.pi/L * m
    print(nu)
    tau_d = 1 / ( k**2 * nu )
    tau_nl = 1 / (k * u_k )
    print('tau_d:', tau_d)
    print('tau_nl:', tau_nl)


    title = fr'L = {L},    N = {N},    dt = {dt:.1e}    $\nu$ = {nu},' + '\n' + fr'N_step = {N_step:.0e},    $\tau_o$ =  {tau_nl:.2f},    $\tau_d$ =  {tau_d:.2f}   '
    fig, axs = plot_template(2,1, figsize = (6, 9))
    plt.rc('font', **{'size'   : 15})
    plt.suptitle(title, fontsize = 15)

    ax = axs[0]
    ax.plot(x, u_init, label = 't=0')
    ax.scatter(x, u_init, s = 3)

    ax = axs[1]
    ax.scatter(k_u_init, u_k2_init, label = 't=0', s=8)

    for j in range(intermediate_step):
        for i in range(int(N_step/intermediate_step)):# Evolution in time
            u_t = int_method(u_t, F, dt, order, nu, dx, 
                             d1,  # first derivative
                             d2  # second derivative
                             )
        tt = (j+1)*int(N_step/intermediate_step)
        u_k2, k_u = evaluate_energy_density_spectrum(u_t.real, dx) # FFT of final u 
        ax = axs[0]
        ax.plot(x, u_t, label = f't={dt*tt}', lw = 0.7)
        ax.scatter(x, u_t.real, s = 5)
        ax = axs[1]
        ax.scatter(k_u, u_k2, label = f't={dt*tt}', s = 8)
    
#    plt.tight_layout()
    plt.legend()
    #plt.savefig(f'../figures/final/burger_t_nl_sim_t_d.png', dpi = 150)
    plt.show()
    
if __name__ == '__main__':
    der_method = 'dfs'
    ampl2(Int.RKN, 4, u_sin_simple, F_burger, der_method, intermediate_step=2)
