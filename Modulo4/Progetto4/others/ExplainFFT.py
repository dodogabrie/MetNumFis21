"""
This file test the Fast Fourier Transform by scipy.
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def my_fft(u, dx):
    """
    Derivative using Fast Fourier Transform.
    """
    dy =  dx / (2 * np.pi) # sampling rate
    fft = fftpack.fft(u) # Discret fourier transform 
    k = fftpack.fftfreq(N, dy) 
    return fft, k

##############################################################

L = 10
N = 100
nu = 0.01

#define the dicrete interval dx
dx = L/N
x = np.linspace(0, L, N, endpoint = False)

###############################################################
#evaluate the function u 
def u(x):
    return np.sin(2*np.pi*x/L)

##########################DERIVATIVE############################
#analitic derivative

def u1(x):
    return 2*np.pi/L * np.cos(2*np.pi*x/L)


der, freq = my_fft(u(x), dx)
plt.scatter(freq, np.abs(der))
plt.show()
