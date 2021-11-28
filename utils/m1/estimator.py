import numpy as np
from numba import njit

# Suscettività magnetica
@njit(fastmath = True)
def compute_chi(m, param):
    L, beta = param
    beta = 1 # Comment here
    return beta * L*L * ( np.mean(m**2) - np.mean(m)**2 )

# Calore specifico
@njit(fastmath = True)
def compute_c(e, param):
    L, beta = param
    kb = 1 # 1.380649 * 1e-23 # J/K --> True value
    beta = 1 # Comment here
    return kb * beta**2 * L*L * ( np.mean(e**2) - np.mean(e)**2)

# Cumulante di Binder
@njit(fastmath = True)
def compute_B(M, param = ()):
    return np.mean(M**4)/(np.mean(M**2)**2)


