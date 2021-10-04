import ising
import numpy as np
import time 
from numpy.random import default_rng
from joblib import Parallel, delayed

def err(X):
    return np.sqrt(1/float(len(X))*1/(float(len(X))-1) * np.sum((X-np.mean(X))**2))

iflag = 0
nlat = 10
beta = 0.3
measures = int(1e4)
i_decorrel = 100
extfield = 0.
njobs = 1

if njobs > 1:
    list_args = [[nlat, iflag, measures, i_decorrel, extfield, beta, i] for i in range(njobs)]
    start = time.time()
    results = Parallel(n_jobs=njobs)(delayed(ising.do_calc)(*arguments) for arguments in list_args)
    print(time.time()-start)
    
    mean_m, mean_e = [], []
    err_m, err_e = [], []

    for m, e in results:
        mean_m.append(np.mean(m))
        mean_e.append(np.mean(e))
        err_m.append(err(m))
        err_e.append(err(e))
    
    mean_e = np.mean(np.array(mean_e))
    mean_m = np.mean(np.array(mean_m))
    err_e = np.mean(np.array(err_e))/np.sqrt(njobs)
    err_m = np.mean(np.array(err_m))/np.sqrt(njobs)
else:
    start = time.time()
    magn, ene = ising.do_calc(nlat, iflag, measures, i_decorrel, extfield, beta, 0)
    print(time.time()-start)
    mean_e = np.mean(ene)
    mean_m = np.mean(magn)
    err_e = err(ene)
    err_m = err(magn)

print(f'Energy: {mean_e:.6f} +- {err_e:.6f}')
print(f'Magn: {mean_m:.6f} +- {err_m:.6f}')
