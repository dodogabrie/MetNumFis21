""" 
This module loop on beta values in the Ising simulation computing M and E
at fixed L.
It also evaluate chi and c for each simulation.
"""
### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

from m1.error import err_mean_corr, err_naive, bootstrap_corr
from estimator import compute_chi, compute_c
import numpy as np
import ising
import time 
from numba import njit
import joblib
from joblib import Parallel, delayed
###

def beta_loop(iflag, nlat, beta_array, 
              measures, i_decorrel, 
              extfield, M, n_jobs):
    def parallel_job(i, beta):
        print(f'L: {nlat}, step {i} su {len(beta_array)}')
        magn, ene = ising.do_calc(nlat, iflag, measures, i_decorrel, extfield, beta)
        m_abs = np.abs(magn)
        chi = compute_chi(m_abs, (nlat, beta))
        c = compute_c(ene, (nlat, beta))
        dchi = bootstrap_corr(m_abs, M, compute_chi, param = (nlat, beta))
        dc = bootstrap_corr(ene  , M, compute_c  , param = (nlat, beta))
        m_abs_mean = np.mean(m_abs)
        dm_abs = err_naive(m_abs)
        ene_mean = np.mean(ene)
        dene = err_naive(ene)
        return [beta, m_abs_mean, dm_abs, ene_mean, dene, chi, dchi, c, dc] 

    list_args = [[i, beta] for i, beta in enumerate(beta_array)]
    list_outputs = Parallel(n_jobs=n_jobs)(delayed(parallel_job)(*args) for args in list_args)

    np.savetxt(f'data/data_obs_nlat{nlat}_test.dat', np.array(list_outputs)) 

    return 

def L_loop(iflag, L_array, beta_array, 
           measures, i_decorrel, 
           extfield, M, njobs = 1):
    for nlat in L_array:
        beta_loop(iflag, nlat, beta_array, 
              measures, i_decorrel, 
              extfield, M, njobs)
    return

if __name__ == '__main__':
    iflag = 1
    beta_array = np.linspace(0.36, 0.48, 30)
    L_array    = np.arange(10, 20, 10, dtype = int)
    measures = int(1e5)
    i_decorrel = 100
    extfield = 0.
    M = 2000
    start = time.time()
    L_loop(iflag, L_array, beta_array, measures, i_decorrel, extfield, M, njobs = 8)
    print('\n', time.time()-start)
