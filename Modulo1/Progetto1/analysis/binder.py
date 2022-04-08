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
from scipy.optimize import curve_fit

def binder_simulation(L_array, param, beta, n_jobs, save_data = True):
    """
    MC simulation of the ising system varying beta, evaluate and write in file
    the Binder cumulant.
    """
    def parallel_job(i, L, param):
        print(f'beta = {beta}, L: {L}')
        if not exists(f'../data/binder_history_beta{beta}/'):
            makedirs(f'../data/binder_history_beta{beta}/')
        magn, ene = ising.do_calc(L, *param, beta, save_data = False)
        np.savetxt(f'../data/binder_history_beta{beta}/nlat{L}.dat', np.column_stack((magn, ene)))
        m_abs = np.abs(magn)
        B = compute_B(magn)
        dB = bootstrap_corr(magn, compute_B)
        return [L, B, dB]

    list_args = [[i, L, param] for i, L in enumerate(L_array)]
    list_outputs = Parallel(n_jobs=n_jobs)(delayed(parallel_job)(*args) for args in list_args)

    if save_data:
        np.savetxt(f'../data/binder_computed_beta{beta}.dat', np.array(list_outputs))
    return

def binder_evaluation(beta):
    """
    Extract the Binder cumulant from a set of montecarlo samplings.
    """
    def extract_B_from_data(data_name):
        # Extract L from file name
        L = int(data_name[4:][:-4]) # Remove 'nlat' and .dat, then turn to float

        # Load all the magnetization and energy at fixed beta
        data = fastload((data_dir + data_name).encode('UTF-8'), int(1e5))
        magn, ene = data[:, 0], data[:, 1]

        print(f'beta = {beta}, L: {L}', end = '\r')
        B = compute_B(magn)
        dB = bootstrap_corr(magn, compute_B)
        return [L, B, dB]

    data_dir = f"../data/binder_history_beta{beta}/"
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    list_outputs = [extract_B_from_data(data_name) for data_name in onlyfiles]

    data = np.array(list_outputs)
    data = data[np.argsort(data[:, 0])]

    np.savetxt(f'../data/binder_computed_beta{beta}.dat', data)
    return

def plot_binder(beta, ax = None):
    """
    Plot the binder cumulant against L, the behaviour should be ~1/L^2
    """
    L, B, dB = np.loadtxt(f'../data/binder_computed_beta{beta}.dat', unpack = True)
    import matplotlib.pyplot as plt
    if ax == None: fig, ax = plt.subplots()
    ax.errorbar(L, B, yerr = dB, color = 'brown', fmt = '.', ms = 4)
    ax.set_xlabel('L', fontsize = 12)
    ax.set_ylabel('B', fontsize = 12)
    #ax.set_title(rf'Cumulante di Binder per $\beta$ = {beta}', fontsize = 15)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    ax.grid(alpha = 0.2)
    return

def fit_binder(beta, remove = 10, ax = None):
    """
    Fit of the vinder cumulant against x = 1/L. This should show a quadratic
    relation between B and x,
    """
    L, B, dB = np.loadtxt(f'../data/binder_computed_beta{beta}.dat', unpack = True)

    L_fit = L[remove:]
    L_outlier = L[:remove]

    B_fit = B[remove:]
    B_outlier= B[:remove]

    dB_fit = dB[remove:]
    dB_outlier = dB[:remove]

    x_fit = 1/L_fit
    x_outlier = 1/L_outlier
    def fit_func(x, a, b):
        return a + b * x * x
    init = (3,-50)
    import matplotlib.pyplot as plt
    if ax == None:
        fig, axs = plt.subplots(2, 1, figsize = (7,6))
    xx = np.linspace(np.min(1/L), np.max(1/L), 1000)
#    plt.plot(xx, fit_func(xx, *init))
    sigma = dB_fit
    w = 1/sigma**2
    pars, covm = curve_fit(fit_func, x_fit, B_fit, init, sigma, absolute_sigma=False)
    chi =((w*(B_fit-fit_func(x_fit,*pars))**2)).sum()
    ndof = len(x_fit) - len(init)
    errors = np.sqrt(np.diag(covm))
    print('Results:')
    a, da = pars[0], errors[0]
    b, db = pars[1], errors[1]
    print(f'a = {a} +- {da}')
    print(f'b = {b} +- {db}')
    print(f'chi = {chi} on {ndof} ndof')
    print(f'chi/ndof = {chi/ndof}')
    # Plot 1
    axs[0].errorbar(L, B, yerr = dB, ls='', fmt ='.', c = 'black')

    axs[0].set_xlabel('L')
    axs[0].set_ylabel('B')
    axs[0].grid(alpha = 0.2)

    # Plot 2: FIT
    axs[1].plot(xx, fit_func(xx, *pars), c = 'blue', lw = 1, label = 'funzione di fit')
    axs[1].errorbar(x_fit, B_fit, yerr = dB_fit, ls='', fmt ='.', c = 'black', label = 'dati di fit')
    axs[1].errorbar(x_outlier, B_outlier, yerr = dB_outlier, ls='', fmt ='.', c = 'red', label = 'outliers')

    axs[1].text(0.8, 0.15, f'B = {a:.6f} $\pm$ {da:.6f}\n$\chi$ ridotto = {(chi/ndof):.2f}',
                horizontalalignment='center', verticalalignment='center',
                transform=axs[1].transAxes, fontsize = 10)
    axs[1].set_xlabel('1/L')
    axs[1].set_ylabel('B')
    axs[1].grid(alpha = 0.2)
    axs[1].legend(loc = 'upper left')
#    axs[1].set_xscale('log')

    plt.suptitle(rf'Cumulante di Binder per $\beta$ = {beta} (fase ordinata)', fontsize = 14)
    plt.tight_layout()
    plt.savefig(f'../figures/Binder_fit/fit_binder{beta}.png', dpi = 200)
    np.savetxt(f'../figures/Binder_fit/fit_binder{beta}.txt', L_fit,
               header = f'L values for binder fit with beta: {beta}', fmt='%.0f')
    plt.show()
    return

if __name__ == '__main__':
    beta_disordered = 0.30
    beta_medium1 = 0.42
    beta_medium2 = 0.46
    beta_ordered = 0.50
    # Simulate the system ranging L for some betas in order to compute Binder
    #%%% Parameters of simulation %%%%%%%%%%%%%%%%%%%%%%%%
#    iflag = 0 # Start hot or cold
#    i_decorrel = 50 # Number of decorrelation for metro
#    extfield = 0. # External field
#    measures = int(1e5) # Number of measures
#    M = 2000 # Blocksize for bootstrap
#    L_min = 5  # Minumu L
#    L_max = 81 # Maximum L
#    L_step = 1 # Step between one L and another
#    n_jobs = 6 # Number of jobs
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    param = (iflag, measures, i_decorrel, extfield)
#    L_array = np.arange(L_min, L_max, L_step)
#    binder_simulation(L_array, param, beta_disordered, M, n_jobs)
#    binder_simulation(L_array, param, beta_ordered, M, n_jobs, save_data=False)

    # Compute Binder cumulant for the simulations runned above
#    beta = beta_disordered
#    binder_evaluation(beta)

#    # Plotting binder vs L
#    import matplotlib.pyplot as plt
#    fig, ax = plt.subplots()
#    plot_binder(beta_disordered, ax)
#    plot_binder(beta_medium1, ax)
#    plt.show()
#    fig, ax = plt.subplots()
#    plot_binder(beta_medium2, ax)
#    plot_binder(beta_ordered, ax)
#    plt.show()

    # Fit binder
#    fit_binder(beta_ordered, remove = 30)
    pass

