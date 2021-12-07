import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import NumDerivative as der

def euler_step(u, F, dt, *params):
    """
    Euler temporal step.
    """
    return u + dt * F(u, *params)

def F(u, c, dx):
    """
    Function of equation:
         du/dt = c * du/dx = F(u)
    """
    return c * der.fft_der(u, dx)

def u(x, L):
    """
    Initial condition (given by a function 'cause here evolve a function.
    """
    k = 1
    return np.sin(k * 2*np.pi*x/L)

#define input variable
# spatial inputs
L = 10 # Spatial size of the grid
N = 100 # spatial step of grid
nu = 0.01 # useless parameter

# temporal inputs
dt = 0.1 # Temporal step
N_step = 50 # Number of Temporal steps

###############################################################
#define the dicrete interval dx
dx = L/N # step size of grid
x = np.linspace(0,L,N, endpoint = False)

speed = dx/dt # speed given the values
c = speed
print(c)

# Define starting function
u_init = u(x, L) # save initial condition 
u_t = np.copy(u_init) # create e copy to evolve it in time

###############################################################
# Evolution varying c, if c > dx/dt the evolution is unstable

#facs = np.linspace(0.1, 3, 3) # array of factors
#for fac in facs:
#    c = speed*fac
#    for i in range(N_step): # temporal evolution
#        u_t = euler_step(u_t, F, dt, c, dx)
#    plt.plot(x, u_t, label = f'final c * {fac:.1f}')
#plt.plot(x, u_init, label = 'init')
#plt.legend()
#plt.xlabel('x')
#plt.ylabel('u')
#plt.show()

###############################################################
# Fourier transform in time 
c = c/2

def my_real_fft(u, dx):
    """
    Real Fast Fourier Transform.
    """
    # x : 2 * pi = y : 1 ---> unitary rate
    # => dy = dx/(2*pi)
    dy =  dx / (2 * np.pi)
    # fft(j) = (u * exp(-2*pi*i*j*np.arange(n)/n)).sum()
    fft = fftpack.rfft(u) # Discret fourier transform 
    k = fftpack.rfftfreq(N, dy) 
    return fft, k

for i in range(N_step):
    u_t = euler_step(u_t, F, dt, c, dx)
fft_u_init, k_u_init = my_real_fft(u_init, dx)
fft_u, k_u = my_real_fft(u_t, dx)

#plt.plot(x, u_init, label = 'init')
#plt.plot(x, u_t, label = 'final')
plt.scatter(k_u, fft_u**2, label = 'final')
plt.scatter(k_u_init, fft_u_init**2, label = 'init')
plt.legend()
plt.yscale('log')
plt.show()
