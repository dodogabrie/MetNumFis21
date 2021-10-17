### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

from m1.error import err_mean_corr, err_naive, bootstrap_corr
import numpy as np
import time 
from numba import njit
import m1.readfile as rf

@njit(fastmath = True)
def estimator(x):
    return np.mean(x**4)/(3*np.mean(x**2)**2)

def test():    
    # Parameters of the test:
    save_results = True # if save the final data
    dir_data = b"data/data.dat" # MC history data
    cut = 2000 # Number of data to cut

    start = time.time()
    data = rf.fastload(dir_data, int(1e7 + 2000))
    print(f'Data imported in {(time.time()-start):.2f}s')
    arr, acc = data[:, 0], data[:, 1]
    print(data[:10])

    arr = arr[cut:]
    
    Ed = estimator(arr)
    #start = time.time()
    Kmax = 12
    err_Ed = np.empty(Kmax)
    start = time.time()
    for k in range(Kmax):
        print(k, end='\r')
        M = int(2**k)
        err_Ed[k] = bootstrap_corr(arr, M, estimator)
    print(f'Time for evaluate errors: {time.time()-start}')
    
    import matplotlib.pyplot as plt
    plt.scatter(np.arange(Kmax), err_Ed)
    plt.xlabel('x')
    plt.ylabel('sigma')
    plt.yscale('log')
    plt.show()
    
    if save_results:
        np.savetxt('bootstrap_testing.txt', np.column_stack((2**np.arange(Kmax), err_Ed)))
    
    print(f'Ed (corr): {Ed} +- {err_Ed[-1]}')


if __name__ == '__main__':
    test()
