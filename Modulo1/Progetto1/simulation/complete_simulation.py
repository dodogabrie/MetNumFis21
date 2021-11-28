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
              extfield, n_jobs):
    def parallel_job(i, beta):
        print(f'L: {nlat}, step {i} su {len(beta_array)}')
        magn, ene = ising.do_calc(nlat, iflag, measures, i_decorrel, extfield, beta, save_data = True, save_lattice = False)
        m_abs = np.abs(magn)
        chi = compute_chi(m_abs, (nlat, beta))
        c = compute_c(ene, (nlat, beta))
        dchi = bootstrap_corr(m_abs, compute_chi, param = (nlat, beta))
        dc = bootstrap_corr(ene  , compute_c  , param = (nlat, beta))
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
           extfield, njobs = 1):
    for i, nlat in enumerate(L_array):
        start = time.time()
        beta_loop(iflag, nlat, beta_array,
              measures, i_decorrel,
              extfield, njobs)
        print(f'\n--> Done L = {nlat} in {time.time()-start}s')
    return

if __name__ == '__main__':

    # GENERAL SIMULATIONS
    #%%% Parameters of simulation %%%%%%%%%%%%%%%%%%%%%%%%
    lock_simulation = True # don't risk to run unwanted simulations
    iflag = 1 # Start hot or cold
    i_decorrel = 50 # Number of decorrelation for metro
    extfield = 0. # External field
    measures = int(1e5) # Number of measures
    beta_min = 0.38 # Minimum value of beta explored
    beta_max = 0.48 # Maximum value of beta explored
    beta_N = 100 # Number of beta observed
    L_min = 10 # Minumu L
    L_max = 80 # Maximum L
    L_step = 10 # Step between one L and another
    njobs = 8 # Number of jobs
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #### Create non-linear beta interval (see explain_bootstrap)###############
    init_array = np.linspace(-1.2, 1.2, beta_N) # start from linear
    trans = np.tan(init_array) # Take tan of linear
    # Normalize the tan in 0-1, then dilatate and traslate in beta_min-beta_max
    trans = (trans + np.abs(np.min(trans)) )/(np.max(trans) - np.min(trans))
    # Finally the beta array
    beta_array = trans * (beta_max - beta_min) + beta_min
    ###########################################################################
    # Create L interval
    L_array    = np.arange(L_min, L_max + L_step, L_step, dtype = int)
    # Starting simulation
    start = time.time()
    if not lock_simulation:
        L_loop(iflag, L_array, beta_array, measures, i_decorrel, extfield, njobs = njobs)
        print('\n########################### Total Time:', time.time()-start)
    if lock_simulation:
        print('The simulation is locked, please change the parameter lock_simulation')


    # PEAK SIMULATIONS
    #%%% Different parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    lock_simulation = True # don't risk to run unwanted simulations
    beta_min = 0.43 # Minimum value of beta explored
    beta_max = 0.44 # Maximum value of beta explored
    beta_N = 24 # Number of beta observed
    L_min = 65 # Minumu L
    L_max = 85 # Maximum L
    L_step = 10 # Step between one L and another
    njobs = 6 # Number of jobs
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    beta_array = np.linspace(beta_min, beta_max, beta_N)
    L_array = np.arange(L_min, L_max + L_step, L_step, dtype = int)
    if not lock_simulation:
        L_loop(iflag, L_array, beta_array, measures, i_decorrel, extfield, njobs = njobs)
        print('\n########################### Total Time:', time.time()-start)
    if lock_simulation:
        print('The simulation is locked, please change the parameter lock_simulation')
