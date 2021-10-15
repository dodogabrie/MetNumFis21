import numpy as np
from numba import jit_module

def bootstrap(arr, M, estimator):
    N = len(arr)
    rest = N%M
    arr_new = np.empty(N-rest)
    fake_estimator = np.empty(M)
    xx = np.arange(M)
    k = int(N/M)
    for j in range(M):
        for i in range(k):
            N_rnd = int(np.random.rand()*k)
            arr_new[i*M : (i+1)*M] = arr[ N_rnd*M : (N_rnd + 1) * M ]
        fake_estimator[j] = estimator(arr_new)
    error = np.std(fake_estimator)
    return error

jit_module(nopython=True, fastmath=True)
