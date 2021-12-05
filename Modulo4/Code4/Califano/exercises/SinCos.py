"""
This file evaluate the derivative of the sin function using varius methods.
"""


#import sys
#sys.path.append('./')

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from Tridiag.tridiag import solve

###############################################################
#define input variable
L = 10
N = 100
nu = 0.01

###############################################################
#define the dicrete interval dx
dx = L/N
x = np.linspace(0,L,N, endpoint = False)

###############################################################
#evaluate the function u 
def u(x):
    return np.sin(2*np.pi*x/L)

##########################DERIVATIVE############################
#analitic derivative

def u1(x):
    return 2*np.pi/L * np.cos(2*np.pi*x/L)

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
    # x : 2 * pi = y : 1 ---> unitary rate
    # => dy = dx/(2*pi)
    dy =  dx / (2 * np.pi)
    # fft(j) = (u * exp(-2*pi*i*j*np.arange(n)/n)).sum()
    fft = fftpack.fft(u) # Discret fourier transform 
    k = fftpack.fftfreq( N, dy) 
    ikfft = 1j * k * fft
    der = fftpack.ifft(ikfft)
    return der.real

def diff_fin_comp_der(u, dx):

    gamma = -1.
    beta = 1.
    alpha = 1.

    n = len(u)

    diag = 4 * np.ones(n)
    diag[0] = diag[0] - gamma
    diag[-1] = diag[-1] - alpha * beta / gamma

    dlo = np.ones(n-1)
    dup = np.ones(n-1)

    b = np.empty(n)
    b[1:-1] = 3/dx * (u[2:] - u[:-2])
    b[0] = 3/dx * (u[1] - u[-1])
    b[-1] = 3/dx * (u[0] - u[-2])

    U = np.zeros(n)
    U[0] = gamma
    U[-1] = alpha

    y, inv = solve(diag, dlo, dup, b)
    z, inv = solve(diag, dlo, dup, U)
    num =  y[0] + beta * y[-1]/gamma
    den = 1 + z[0] + beta * z[-1]/gamma
    der = y - num/den * z
    return der
########################SAVE IN FILE############################


#####save results in file txt
np.savetxt('SinCos.txt', np.column_stack((x,u(x),u1(x))),
            header = 'x    u   u1', fmt = '%.4f')


#######################VERIFY USING PLOT#########################
#verify periodicity of function u
#plt.plot(np.concatenate((x,x+L)),np.concatenate((u(x), u(x+L))))
#plt.show()

#verify periodicity of derivative
#plt.plot(np.concatenate((x,x+L)),np.concatenate((backward_der(u(x), dx), backward_der( u(x+L), dx ))))
#plt.show()

#plot function u and derivative
#plt.plot(x,u(x))
#plt.plot(x,u1(x))
#plt.show()

#####################NUMERICAL VS ANALITICAL######################
#Compare numerical derivative and analitical derivative
plt.plot(x, u1(x), color = 'blue')
plt.plot(x, forward_der(u(x), dx), color = 'red', label = 'forw')
plt.plot(x, backward_der(u(x), dx), color = 'green', label = 'backw')
plt.plot(x, simm_der(u(x), dx), color = 'pink', label = 'simm')
plt.plot(x, fft_der(u(x), dx), color = 'black', label = 'fft')
plt.plot(x, diff_fin_comp_der(u(x), dx), color = 'brown', label = 'fft')
plt.legend()
plt.show()
