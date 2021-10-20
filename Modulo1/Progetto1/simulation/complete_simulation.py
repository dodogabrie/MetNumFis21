""" 
This module loop on beta values and on L values in the Ising simulation 
computing M and E. It also evaluate chi and c (and the relative errors) 
for each simulation.
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

from m1.error import err_mean_corr, err_naive, bootstrap_corr
from m1.estimator import compute_chi, compute_c
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
        print(f'L: {nlat}, step {i} su {len(beta_array)}', end = '\r')
        magn, ene = ising.do_calc(nlat, iflag, measures, i_decorrel, extfield, beta, save_data = True)
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

    np.savetxt(f'../data/data_obs_nlat{nlat}_test_new.dat', np.array(list_outputs)) 

    return 

def L_loop(iflag, L_array, beta_array, 
           measures, i_decorrel, 
           extfield, M, njobs = 1):
    for i, nlat in enumerate(L_array):
        start = time.time()
        beta_loop(iflag, nlat, beta_array, 
              measures, i_decorrel, 
              extfield, M, njobs)
        print(f'\n--> Done L = {nlat} in {time.time()-start}s')
    return

if __name__ == '__main__':
    #%%% Parameters of simulation %%%%%%%%%%%%%%%%%%%%%%%%
    iflag = 1 # Start hot or cold
    i_decorrel = 50 # Number of decorrelation for metro
    extfield = 0. # External field
    measures = int(1e5) # Number of measures
    M = 2000 # Blocksize for bootstrap
    beta_min = 0.38 # Minimum value of beta explored
    beta_max = 0.48 # Maximum value of beta explored
    beta_N = 100 # Number of beta observed
    njobs = 8
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create non-linear beta interval (see explain_bootstrap)
    init_array = np.linspace(-1.2, 1.2, beta_N) # start from linear
    trans = np.tan(init_array) # Take tan of linear
    # Normalize the tan in 0-1, then dilatate and traslate in beta_min-beta_max
    trans = (trans + np.abs(np.min(trans)) )/(np.max(trans) - np.min(trans))
    # Finally the beta array
    beta_array = trans * (beta_max - beta_min) + beta_min 
    L_array    = np.arange(10, 100, 10, dtype = int)
    start = time.time()
    L_loop(iflag, L_array, beta_array, measures, i_decorrel, extfield, M, njobs = njobs)
    print('\n########################### Total Time:', time.time()-start)
