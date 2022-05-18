import numpy as np
from numba import jit_module


def err_mean_corr(x, kmax = None):
    """
    Error on the mean of sampling by a Montecarlo simulation
    considering the correlation between the point i and
    the point i + k for k in [1, kmax].

    Parameters
    ----------
    x : numpy 1d array
        Output sample of the MC.
    kmax: int
        Maximum correlation searched (lower or equal to len(x)).
        If None kmax evaluating when Ck goes under 1/e.

    Returns
    -------
    (float, float, numpy 1d array)
        (tau, error with correlation, Array of Ck)
    """
    if kmax == None:
        Ck = _C_early(x)
        tau = len(Ck)
        if tau == 1: tau = 0
    else:
        Ck = _C(x, kmax)
        tau = np.sum(Ck)
    return tau, err_naive(x) * np.sqrt(1 + 2*tau), Ck

## AUX function for err_mean_corr ####################################################
def _PairCorr(x, y, mean_x):
    return (x - mean_x) * (y - mean_x)

def _C(x, kmax):
    e = np.exp(1)
    N = len(x)
    Ck = np.empty(kmax-1)
    mean_x = np.mean(x)
    sigma2_inv = 1/np.mean(_PairCorr(x,x, mean_x))
    for k in range(kmax-1):
        Ck[k] = sigma2_inv * np.mean(_PairCorr(x[0 : N-k ], x[ k : N ], mean_x))
    return Ck

def _C_early(x):
    e = np.exp(1)
    N = len(x)
    kmax = N
    Ck = np.empty(kmax-1)
    mean_x = np.mean(x)
    sigma2_inv = 1/np.mean(_PairCorr(x,x, mean_x))
    for k in range(kmax-1):
        Ck[k] = sigma2_inv * np.mean(_PairCorr(x[0 : N-k ], x[ k : N ], mean_x))
        if Ck[k] <= 1/e:
            break
    return Ck[:k]
######################################################################################

def err_naive(X):
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

def bootstrap_corr(arr, estimator, n_fake_samples = 200, param = ()):
    N = len(arr) # Number of data in initial sample
    kk = np.arange(4, int(np.log2(N)))
    list_std = np.empty(len(kk))
    for b in range(len(kk)):
        M = 2**kk[b]
        rest = N % M # Number of data not considered
        arr_new = np.empty((N - rest, arr.shape[-1])) # Array divisible by M
        fake_estimator = np.empty( n_fake_samples ) # Initialized array for fake estimators
        n_block = int(N/M) # Number of blocks
        for j in range(n_fake_samples): # loop over number of fake samples
            N_rnd = (np.random.rand(n_block)*n_block).astype(np.int_) # Random number in [0, n_block]
            for i in range(n_block): # loop over the number of blocks (of each fake samples)
                # Core of the bootstrap: create fake sample using numpy slicing
                arr_new[i*M : (i+1)*M] = arr[ N_rnd[i]*M : (N_rnd[i] + 1) * M ]
            fake_estimator[j] = estimator(arr_new, param) # Computing fake estimator for fake sample j
        error = np.std(fake_estimator) # Evaluate the standard dev over all fake samples
        list_std[b] = error
    return np.max(list_std) # :)

jit_module(nopython = True, fastmath = True, cache = True)
