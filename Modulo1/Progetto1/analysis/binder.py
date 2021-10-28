"""
Analysis of the Binder cumulant: simulate the system for other values of L and
in order to have an accurate estimation of B.
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
sys.path.append('../simulation/')
####################################################################


import numpy as np
from m1.error import err_mean_corr, err_naive, bootstrap_corr
from m1.readfile import slowload, fastload
from m1.estimator import compute_B
from os import listdir, makedirs
from os.path import isfile, join, exists
import joblib
from joblib import Parallel, delayed
import ising
import time

def binder_simulation(L_array, param, beta, M, n_jobs):
    def parallel_job(i, L, param):
        print(f'beta = {beta}, L: {L}')
        if not exists(f'../data/binder_history_beta{beta}/'):
            makedirs(f'../data/binder_history_beta{beta}/')
        magn, ene = ising.do_calc(L, *param, beta, save_data = False)
        np.savetxt(f'../data/binder_history_beta{beta}/nlat{L}.dat', np.column_stack((magn, ene)))
        m_abs = np.abs(magn)
        B = compute_B(magn)
        dB = bootstrap_corr(magn, M, compute_B)
        return [L, B, dB]

    list_args = [[i, L, param] for i, L in enumerate(L_array)]
    list_outputs = Parallel(n_jobs=n_jobs)(delayed(parallel_job)(*args) for args in list_args)

    np.savetxt(f'../data/binder_computed_beta{beta}.dat', np.array(list_outputs))
    return

def binder_evaluation(beta, M):
    def extract_B_from_data(data_name):
        # Extract L from file name
        L = int(data_name[4:][:-4]) # Remove 'nlat' and .dat, then turn to float

        # Load all the magnetization and energy at fixed beta
        data = fastload((data_dir + data_name).encode('UTF-8'), int(1e5))
        magn, ene = data[:, 0], data[:, 1]

        print(f'beta = {beta}, L: {L}', end = '\r')
        B = compute_B(magn)
        dB = bootstrap_corr(magn, M, compute_B)
        return [L, B, dB]

    data_dir = f"../data/binder_history_beta{beta}/"
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    list_outputs = [extract_B_from_data(data_name) for data_name in onlyfiles]

    data = np.array(list_outputs)
    data = data[np.argsort(data[:, 0])]

    np.savetxt(f'../data/binder_computed_beta{beta}.dat', data)
    return

def plot_binder(beta):
    L, B, dB = np.loadtxt(f'../data/binder_computed_beta{beta}.dat', unpack = True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.errorbar(L, B, yerr = dB, color = 'brown', fmt = '.', ms = 4)
    ax.set_xlabel('L', fontsize = 12)
    ax.set_ylabel('B', fontsize = 12)
    ax.set_title(rf'Cumulante di Binder per $\beta$ = {beta}', fontsize = 15)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    ax.grid(alpha = 0.2)
    plt.show()
    return

if __name__ == '__main__':
    # Simulate the system ranging L for some betas in order to compute Binder
    #%%% Parameters of simulation %%%%%%%%%%%%%%%%%%%%%%%%
    lock_simulation = True # don't risk to run unwanted simulations
    iflag = 1 # Start hot or cold
    i_decorrel = 50 # Number of decorrelation for metro
    extfield = 0. # External field
    measures = int(1e5) # Number of measures
    M = 2000 # Blocksize for bootstrap
    L_min = 5  # Minumu L
    L_max = 81 # Maximum L
    L_step = 1 # Step between one L and another
    n_jobs = 6 # Number of jobs
    beta_disordered = 0.30
    beta_medium1 = 0.42
    beta_medium2 = 0.46
    beta_ordered = 0.50
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param = (iflag, measures, i_decorrel, extfield)
    L_array = np.arange(L_min, L_max, L_step)
    binder_simulation(L_array, param, beta_disordered, M, n_jobs)
    binder_simulation(L_array, param, beta_medium1, M, n_jobs)
    binder_simulation(L_array, param, beta_medium2, M, n_jobs)
    binder_simulation(L_array, param, beta_ordered, M, n_jobs)

    # Compute Binder cumulant for the simulations runned above
#    beta = 0.38
#    M = 2000
#    binder_evaluation(beta, M)
#
#    # Plotting binder vs L
#    beta = 0.38
#    plot_binder(beta)
    pass

