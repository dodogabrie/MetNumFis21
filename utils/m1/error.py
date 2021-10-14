import numpy as np
from numba import jit_module

def err_corr(x, kmax):
    """
    Error on a Montecarlo simulation considering the correlation 
    between the point i and the point i + k for k in [1, kmax].

    Parameters
    ----------
    x : numpy 1d array
        Output sample of the MC.
    kmax: int
        Maximum correlation searched (lower or equal to len(x)).

    Returns 
    -------
    (float, float, numpy 1d array)
        (tau, error with correlation, Array of Ck)
    """
    Ck = _C(x, kmax)
    tau = np.sum(Ck)
    return tau, err(x) * np.sqrt(1 + 2*tau), Ck
    
def err(X):
    """
    Montecarlo error considering just the variance estimation (so 
    no correlation).
    
    Parameters
    ----------
    x : numpy 1d array
        Output sample of the MC.

    Returns
    -------
    float
        Error without correlation
    """
    N = float(len(X))
    return np.sqrt( 1/N * 1/(N-1) * np.sum((X-np.mean(X))**2))

def _PairCorr(x, y, mean_x):
    return (x - mean_x) * (y - mean_x)

def _C(x, kmax):
    N = len(x)
    Ck = np.empty(kmax-1)
    mean_x = np.mean(x)
    for k in range(kmax-1):
        Ck[k] = 1/(N-(k+1)) * np.sum(_PairCorr(x[0 : N-(k+1) ], x[ k+1 : N ], mean_x))
    return Ck 



#jit_module(nopython = True, fastmath = True, cache = True)
