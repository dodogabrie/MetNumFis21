"""
This file uses the Euler integrator to solve the equation:
    d_t ( u ) = - c * d_x ( F ( u ) )
Where d_t and d_x are the derivative respect t and x.
It use the derivative with the trick suggested by the professor Mannella.
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import NumDerivative as der
import NumIntegrator as Int
import NumScheme as NS

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
    return A * np.cos(k * 2*np.pi*x/L)

def simm_der(u, dx):
    """
    Derivative considering the previous point
    """
    der = np.empty(len(u))
    der[1:-1] = (u[2:] - u[:-2])/(2*dx)
    der[0], der[-1] = der[-2], der[1]
    return der

def test_boundary(u):
    """
    This function is a test for the new boundary condition used by 
    prof. Mannella.
    """
    L = 10 # Spatial size of the grid
    N = 100 # spatial step of grid
    
    # temporal inputs
    dt = 0.1 # Temporal step
    N_step = 2 # Number of Temporal steps
    
    ###############################################################
    #define the dicrete interval dx
    dx = L/N # step size of grid
    x = np.linspace(0, L + 2*dx, N + 2, endpoint = False)
    
    speed = dx/dt # speed given the values
    c = speed
    
    # Define starting function
    u_init = u(x, L) # save initial condition 
    der_u = simm_der(u_init, dx)

    plt.plot(np.concatenate((x[:-2], L + x[:-2])), np.concatenate((u_init[:-2], u_init[:-2])))
    plt.show()
    plt.plot(np.concatenate((x[:-2], L + x[:-2])), np.concatenate((der_u[:-2], der_u[:-2])))
    plt.show()


if __name__ == '__main__':
    test_boundary(u_sin_simple)
