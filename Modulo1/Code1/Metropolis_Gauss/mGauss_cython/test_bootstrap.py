### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

from m1.error import err_corr, err
import numpy as np
import time 
import m1.bootstrap as bs
from numba import njit
import m1.readfile as rf

@njit(fastmath = True)
def estimator(x):
    return np.mean(x**4)/(3*np.mean(x**2)**2)

datafile = b"data/data.dat"
data = rf.fastload(datafile, int(1e7 + 2000))
arr, acc = data[:, 0], data[:, 1]
print(arr[:20])

nstat = len(arr)
cut = 2000
arr = arr[cut:]

x4 = arr**4
x2 = arr**2

print(f'Mean x4 (no corr): {np.mean(x4):.6f} +- {err(x4):.6f}')
print(f'Mean x2 (no corr): {np.mean(x2):.6f} +- {err(x2):.6f}')
Ed = np.mean(x4)/(3*np.mean(x2)**2)
#start = time.time()
Kmax = 8
arr_err = np.empty(Kmax)
for k in range(Kmax):
    print(k, end='\r')
    M = int(2**k)
    err_Ed = bs.bootstrap(arr, M, estimator)
    arr_err[k] = err_Ed

import matplotlib.pyplot as plt
plt.scatter(np.arange(Kmax), arr_err)
plt.xlabel('x')
plt.ylabel('sigma')
plt.yscale('log')
plt.show()

np.savetxt('bootstrap_testing.txt', np.column_stack((2**np.arange(Kmax), arr_err)))


print(f'Ed (corr): {Ed} +- {err_Ed}')
