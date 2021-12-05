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
    der = fftpack.ifft(ikfft)
    return der

def diff_fin_comp_der(u, dx):
    """
    Compact finite different derivative using Shepard Morrisoni formula.
    """

    gamma = -1. # Why this gamma work? Literature say to use -diag[0] = -4
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
