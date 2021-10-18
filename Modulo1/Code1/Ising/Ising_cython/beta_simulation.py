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

def beta_loop(iflag, nlat, beta_array, 
              measures, i_decorrel, 
              extfield, M, njobs):
    # Define quantities
    chi_array = np.empty(len(beta_array))
    dchi_array = np.empty(len(beta_array))
    c_array = np.empty(len(beta_array))
    dc_array = np.empty(len(beta_array))
    m_abs_array = np.empty(len(beta_array))
    dm_abs_array = np.empty(len(beta_array))
    ene_array = np.empty(len(beta_array))
    dene_array = np.empty(len(beta_array))
    ####
    for i, beta in enumerate(beta_array):
        print(f'L: {nlat}, step {i} su {len(beta_array)}', end='\r')
        magn, ene = ising.do_calc(nlat, iflag, measures, i_decorrel, extfield, beta)
        m_abs = np.abs(magn)
        chi_array[i] = compute_chi(m_abs, (nlat, beta))
        c_array[i] = compute_c(ene, (nlat, beta))
        dchi_array[i] = bootstrap_corr(m_abs, M, compute_chi, param = (nlat, beta))
        dc_array[i]   = bootstrap_corr(ene  , M, compute_c  , param = (nlat, beta))
        m_abs_array[i] = np.mean(m_abs)
        dm_abs_array[i] = err_naive(m_abs)
        ene_array[i] = np.mean(ene)
        dene_array[i] = err_naive(ene)
    ####
    np.savetxt(f'data/data_obs_nlat{nlat}.dat', 
               np.column_stack((beta_array, m_abs_array, dm_abs_array, 
                                            ene_array  , dene_array  ,
                                            chi_array  , dchi_array  ,
                                            c_array    , dc_array     )
                              )
               ) 
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
    nlat = 10
    beta = np.linspace(0.36, 0.48, 96)
    L    = np.arange(10, 40, 10, dtype = int)
    measures = int(1e5)
    i_decorrel = 50
    extfield = 0.
    M = 2000
    start = time.time()
#    beta_loop(iflag, nlat, beta, measures, i_decorrel, extfield, M)
    print('\n', time.time()-start)
