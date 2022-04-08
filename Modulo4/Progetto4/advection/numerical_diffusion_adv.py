"""
This file uses the Euler integrator to solve the equation:
    d_t ( u ) = - c * d_x ( F ( u ) )
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
def F(u, c, dx, der_method):
    """
    Function of equation:
         du/dt = - c * du/dx = F(u)
    """
    return - c * der_method(u, dx)

# Define the initial condition
def u_sin_simple(x, L, k = 1):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    return np.sin(k * 2*np.pi*x/L)

def test_numdiff(int_method, u):
    #define input variable ########################################
    # spatial inputs
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    
    # temporal inputs
    dt = 0.01 # Temporal step
    N_step = 5 # Number of Temporal steps
    ###############################################################

    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0,L,N, endpoint = False)
    
    speed = dx/dt # speed given the values
    c = speed/2

    # generate initial condition with all the m
    list_m = np.arange(1, 25)
    u_init_all_k = 0
    for m in list_m:
        u_init_all_k += u(x, L, k = m)
    u_init = np.copy(u_init_all_k)

    
    data_to_save_general = np.column_stack((x, u_init))

    list_nu = np.array([1e-2, 1e-3, 3e-4, 1e-4])
    list_nu_str = ''
    for nu in list_nu:
        list_nu_str += f'    u_nu{nu:.0e}'

    print(f'final physic time: {N_step*dt}')
    der_methods = [der.simm_der2, der.diff_fin_comp_der2, der.fft_der2]
    file_names = ['dfs.txt', 'dfc.txt', 'fft.txt']

    for filename, der_method in zip(file_names, der_methods):
        data = np.copy(data_to_save_general) # copy x, u_init
        u_t = np.copy(u_init)
        for i in range(N_step):# Evolution in time
            u_t = int_method(u_t, F, dt, 10, c, dx, der_method)
        data = np.column_stack((data, u_t))
        np.savetxt('../data/advection/varying_der/'+filename, data, header='x    u_init' + list_nu_str)
 

if __name__ == '__main__':
    test_numdiff(Int.RKN, u_sin_simple)
