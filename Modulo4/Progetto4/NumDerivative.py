"""
This script contains some methods to evaluate the derivative of a function
given f and the grid step size dx.
"""

import numpy as np
from scipy import fftpack
from Tridiag.tridiag import solve

#numerical derivative
def forward_der(u,dx):
    """
    Derivative considering the next point
    """
    der = np.empty(len(u))
    der[:-1] = (u[1:] - u[:-1])/dx
    der[-1] = (u[0] - u[-1])/dx # periodic boundary condition
    return der

def backward_der(u,dx):
    """
    Derivative considering the previous point
    """
    der = np.empty(len(u))
    der[1:] = (u[1:] - u[:-1])/dx
    der[0] = (u[0] - u[-1])/dx # periodic boundary condition
    return der

def simm_der(u,dx):
    """
    Derivative considering the previous point
    """
    der = np.empty(len(u))
    der[1:-1] = (u[2:] - u[:-2])/(2*dx)
    der[0] = (u[1] - u[-1])/(2*dx) # left periodic boundary condition
    der[-1] = (u[0] - u[-2])/(2*dx) # right periodic boundary condition
    return der

def simm_der_Ord4(u, dx):
    der = np.empty(len(u))
    der[2:-2] = 1/(12 * dx) * (u[:-4] - 8*u[1:-3] + 8*u[3:-1] - u[4:])
    for i in range(-2,2):
        der[i] = 1/(12 * dx) * (u[i-2] - 8*u[i-1] + 8*u[i+1] - u[i+2])
    return der

def simm_der_Ord6(u, dx):
    der = np.empty(len(u))
    der[3:-3] = 1/(60 * dx) * (-u[:-6] + 9*u[1:-5] - 45*u[2:-4] + \
                               45*u[4:-2] - 9*u[5:-1] + u[6:])
    for i in range(-3,3):
        der[i] = 1/(60 * dx) * (-u[i-3] + 9*u[i-2] - 45*u[i-1] + \
                               45*u[i+1] - 9*u[i+2] + u[i+3])
    return der

def fft_der(u,dx):
    """
    Derivative using Fast Fourier Transform.
    """
    N = len(u)
    # x : 2 * pi = y : 1 ---> unitary rate
    # => dy = dx/(2*pi)
    dy =  dx / (2 * np.pi)
    # fft(j) = (u * exp(-2*pi*i*j*np.arange(n)/n)).sum()
    fft = fftpack.fft(u) # Discret fourier transform 
    k = fftpack.fftfreq( N, dy) 
    ikfft = 1j * k * fft
    return fftpack.ifft(ikfft).real
    

def diff_fin_comp_der_magic(u, dx):
    """
    Compact finite different derivative using Shepard Morrisoni formula.
    We did not really understand why this work, is like a magic formula about maths...
    References:
    - http://wwwmayr.in.tum.de/konferenzen/Jass09/courses/2/Soldatenko_presentation.pdf
    - http://www.phys.lsu.edu/classes/fall2013/phys7412/lecture10.pdf
    """

    gamma = -4. # -1 was an error: problem solved in the tridiagonal routine...
    beta = 1. # term out of diagonal (upper right)
    alpha = 1. # term out of diagonal (lower left)

    n = len(u) # size of system

    # define the diagonal elements
    diag = 4 * np.ones(n)
    # Subtract the term for the SM formula
    diag[0] = diag[0] - gamma
    diag[-1] = diag[-1] - alpha * beta / gamma

    # define lower and upper diagonal
    dlo = np.ones(n-1)
    dup = np.ones(n-1)

    # Define right side term
    b = np.empty(n)
    b[1:-1] = 3/dx * (u[2:] - u[:-2])
    b[0] = 3/dx * (u[1] - u[-1]) # periodic conditions
    b[-1] = 3/dx * (u[0] - u[-2])# periodic conditions

    # define the vector for auxiliar system U = [gamma, 0, ..., 0, alpha]
    U = np.zeros(n)
    U[0] = gamma
    U[-1] = alpha

    # solve A * y = b
    y, inv = solve(diag, dlo, dup, b)
    # solve A * z = U
    z, inv = solve(diag, dlo, dup, U)

    # Define the factor for the final derivative
    num =  y[0] + beta * y[-1]/gamma
    den = 1 + z[0] + beta * z[-1]/gamma

    # compute the derivative using the factor.
    der = y - num/den * z
    return der

def diff_fin_comp_der(u, dx):
    """
    Compact finite different derivative using 'splitting tridiagonal idea'.
    This one is clear to us and it is easy to learn (same results that previous,
    is just the same idea with a different implementation).
    References:
    http://www.sciencedirect.com/science/article/pii/0021999175900819/pdf?md5=4b9194f11c72ae3e22efccd620719bf9&pid=1-s2.0-0021999175900819-main.pdf
    """
    # Initialize size of the system
    n = len(u)
    # Initialize results array
    der = np.empty(n)
    
    # Define the tridiagonal system
    diag = 4 * np.ones(n)
    dlo = np.ones(n-1)
    dup = np.ones(n-1)

    b = np.empty(n)
    b[1:-1] = 3/dx * (u[2:] - u[:-2])
    b[0] = 3/dx * (u[1] - u[-1])
    b[-1] = 3/dx * (u[0] - u[-2])

    # Define auxiliar vector
    F = np.zeros(n-1)
    F[0] = 1
    F[-1] = 1

    # E is the tridiagonal matrix but the last line/row:
    # diag[:-1], dlo[:-1], dup[:-1]
    # b_tilde is just b[:-1]
    v, inv = solve(diag[:-1], dlo[:-1], 
                   dup[:-1], b[:-1])
    u, inv = solve(diag[:-1], dlo[:-1], 
                   dup[:-1], F)

    # Return the results using the solution of the system of 2 equations
    der[-1] = (b[-1] - (v[0]+v[-1]))/(diag[0] - (u[0] + u[-1])) # f'_tilde
    der[:-1] = v - u * der[-1] # f'_n
    return der

########################SECOND DERIVATIVE########################

def simm_der2(u, dx):
    """
    Second symmetric derivative
    """
    der = np.empty(len(u))
    der[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2])/(dx**2)
    der[0] = (u[1] - 2 * u[0] + u[-1])/(dx**2) # left periodic boundary condition
    der[-1] = (u[0] - 2 * u[-1] + u[-2])/(dx**2) # right periodic boundary condition
    return der

def simm_der_Ord4_2(u, dx):
    der = np.empty(len(u))
    der[2:-2] = 1/(12 * dx**2) * (-u[:-4] + 16*u[1:-3] - 30*u[2:-2] + 16*u[3:-1] - u[4:])
    for i in range(-2,2):
        der[i] = 1/(12 * dx**2) * (-u[i-2] - 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2])
    return der

def simm_der_Ord6_2(u, dx):
    der = np.empty(len(u))
    der[3:-3] = 1/(180*dx**2)*(2*u[:-6] - 27*u[1:-5] + 270*u[2:-4] - 490*u[3:-3] +\
            270*u[4:-2] - 27*u[5:-1] + 2*u[6:])
    for i in range(-3, 3):
        der[i] = 1/(180*dx**2)*(2*u[i-3] - 27*u[i-2] + 270*u[i-1] - 490*u[i] +\
            270*u[i+1] - 27*u[i+2] + 2*u[i+3])
    return der

def fft_der2(u,dx):
    """
    Second derivative using Fast Fourier Transform.
    """
    N = len(u)
    # x : 2 * pi = y : 1 ---> unitary rate
    # => dy = dx/(2*pi)
    dy =  dx / (2 * np.pi)
    # fft(j) = (u * exp(-2*pi*i*j*np.arange(n)/n)).sum()
    fft = fftpack.fft(u) # Discret fourier transform 
    k = fftpack.fftfreq( N, dy) 
    k2fft = - k**2 * fft
    return fftpack.ifft(k2fft).real

def diff_fin_comp_der2(u, dx):
    """
    Compact finite different second derivative using 'splitting tridiagonal idea'.
    This one is clear to us and it is easy to learn (same results that previous,
    is just the same idea with a different implementation).
    """
    # Initialize size of the system
    n = len(u)
    # Initialize results array
    der = np.empty(n)
    
    # Define the tridiagonal system
    diag = 10 * np.ones(n)
    dlo = np.ones(n-1)
    dup = np.ones(n-1)

    b = np.empty(n)
    b[1:-1] = 12/dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
    b[0] = 12/dx**2 * (u[1] - 2 * u[0] + u[-1])
    b[-1] = 12/dx**2 * (u[0] - 2 * u[-1] + u[-2])

    # Define auxiliar vector
    F = np.zeros(n-1)
    F[0] = 1
    F[-1] = 1

    # E is the tridiagonal matrix but the last line/row:
    # diag[:-1], dlo[:-1], dup[:-1]
    # b_tilde is just b[:-1]
    v, inv = solve(diag[:-1], dlo[:-1], 
                   dup[:-1], b[:-1])
    u, inv = solve(diag[:-1], dlo[:-1], 
                   dup[:-1], F)

    # Return the results using the solution of the system of 2 equations
    der[-1] = (b[-1] - (v[0]+v[-1]))/(diag[0] - (u[0] + u[-1])) # f'_tilde
    der[:-1] = v - u * der[-1] # f'_n
    return der

