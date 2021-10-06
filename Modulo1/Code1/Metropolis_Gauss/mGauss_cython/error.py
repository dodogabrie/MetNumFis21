import numpy as np
from numba import jit_module

def dfi_dfik(x, k, N, mean_x):
    return (x[ 0 : N-k ] - mean_x) * (x[ k :  N  ] - mean_x)

def C(x, kmax):
    N = len(x)
    Ck = np.empty(kmax-1)
    mean_x = np.mean(x)
    for k in range(1, kmax):
        Ck[k-1] = 1/(N-k) * np.sum(dfi_dfik(x, k, N, mean_x))
    return Ck

def err_corr(x, kmax):
    Ck = C(x, kmax)
    tau = np.sum(Ck)
    return tau, err(x) * np.sqrt(1 + 2*tau), Ck
    
def err(X):
    N = float(len(X))
    return np.sqrt( 1/N * 1/(N-1) * np.sum((X-np.mean(X))**2))

jit_module(nopython = True, fastmath = True, cache = True)
