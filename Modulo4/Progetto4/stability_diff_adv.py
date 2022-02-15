import numpy as np 
import matplotlib.pyplot as plt

dx = 0.1
dt = 0.1
nu = 0.1 
c = dx/dt * 0.5
def eq(k, dx, dt, nu, c):
    r_c = c * dt / ( 2 * dx )
    r_nu = nu * dt / ( dx * dx )
    theta = k * dx
    r_c = 
    return 1 + 2 * r_c * np.sin(2 * theta) - 4 * r_nu * np.sin(theta/2)

