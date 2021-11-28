### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

from m1.error import err_mean_corr, err_naive, bootstrap_corr
import ising
import numpy as np
import time 
from numpy.random import default_rng
from joblib import Parallel, delayed

iflag = 1
nlat = 10
beta = 0.3
measures = int(1e2)
i_decorrel = 50
extfield = 0.
njobs = 1

if njobs > 1:
    list_args = [[nlat, iflag, measures, i_decorrel, extfield, beta] for i in range(njobs)]
    start = time.time()
    results = Parallel(n_jobs=njobs)(delayed(ising.do_calc)(*arguments) for arguments in list_args)
    print(time.time()-start)
    
    mean_m, mean_e = [], []
    err_m, err_e = [], []

    for m, e in results:
        mean_m.append(np.mean(m))
        mean_e.append(np.mean(e))
        err_m.append(err_naive(m))
        err_e.append(err_naive(e))
    
    mean_e = np.mean(np.array(mean_e))
    mean_m = np.mean(np.array(mean_m))
    err_e = np.mean(np.array(err_e))/np.sqrt(njobs)
    err_m = np.mean(np.array(err_m))/np.sqrt(njobs)
else:
    start = time.time()
    magn, ene = ising.do_calc(nlat, iflag, measures, i_decorrel, extfield, beta)
    print(time.time()-start)
    mean_e = np.mean(ene)
    mean_m = np.mean(magn)
    err_e = err_naive(ene)
    err_m = err_naive(magn)

print(f'Energy: {mean_e:.6f} +- {err_e:.6f}')
print(f'Magn: {mean_m:.6f} +- {err_m:.6f}')
