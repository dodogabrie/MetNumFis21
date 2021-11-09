"""
Fit to estimate the maximum of the suscettivity (chi_max = chi(beta_max)) and
the relative beta_max point.
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
sys.path.append('../simulation/')
####################################################################

import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


def fit_chi_beta(L, sigma_fraction= 0.5):
    data_file = f'../data/final_results/data_obs_nlat{L}_test_final.dat'
    beta, _, _, _, _, chi, dchi, _, _ = np.loadtxt(data_file, unpack = True)
    # Mean and standard deviation
    chi_max_hypot = np.max(chi)
    chi_dev = np.std(chi)*sigma_fraction
    # extract values on peak
    chi_fit = chi[chi > chi_max_hypot - chi_dev]
    dchi_fit = dchi[chi > chi_max_hypot - chi_dev]
    beta_fit = beta[chi > chi_max_hypot - chi_dev]

    # Fit function
    def fit_func(x, chi_max, x_c, b):
        return chi_max - b*(x - x_c)**2

    # Initial conditions
    init = [np.max(chi_fit), beta_fit[np.argmax(chi_fit)], 10]
    xx = np.linspace(np.min(beta_fit), np.max(beta_fit), 1000)
#    plt.plot(xx, fit_func(xx, *init))
    sigma = dchi_fit
    w = 1/sigma**2
    pars, covm = curve_fit(fit_func, beta_fit, chi_fit, init, sigma, absolute_sigma=False)
    chi =((w*(chi_fit-fit_func(beta_fit,*pars))**2)).sum()
    ndof = len(beta_fit) - len(init)
    errors = np.sqrt(np.diag(covm))
    print('Results:')
    chi_max, dchi_max = pars[0], errors[0]
    beta_max, dbeta_max = pars[1], errors[1]
    b, db = pars[2], errors[2]
    print(f'chi_m = {chi_max} +- {dchi_max}')
    print(f'beta_m = {beta_max} +- {dbeta_max}')
    print(f'b = {b} +- {db}')
    print(f'chi = {chi} on {ndof} ndof')
    print(f'chi/ndof = {chi/ndof}')

    plt.plot(xx, fit_func(xx, *pars))
    plt.errorbar(beta_fit, chi_fit, yerr = dchi_fit, fmt = '.')
#    plt.show()
    return chi_max, dchi_max, beta_max, dbeta_max, chi/ndof

if __name__ == '__main__':
    list_L          = [10,  15,  20,  25,  30,   35,   40,   50,  60,  70,  80]
    sigma_fractions = [1.2, 0.4, 0.4, 0.4, 0.33, 0.5, 0.45, 0.32, 0.4, 0.7, 0.6]
    chi_max_list, dchi_max_list, beta_max_list, dbeta_max_list, chi_red_list = [],[],[],[],[]
    for L, frac in zip(list_L, sigma_fractions):
        chi_max, dchi_max, beta_max, dbeta_max, chi_red = fit_chi_beta(L, frac)
        chi_max_list.append(chi_max)
        dchi_max_list.append(dchi_max)
        beta_max_list.append(beta_max)
        dbeta_max_list.append(dbeta_max)
        chi_red_list.append(chi_red)
    np.savetxt('../data/fit_beta_chi_max.txt',
               np.column_stack((list_L, chi_max_list, dchi_max_list, beta_max_list, dbeta_max_list, chi_red_list)),
               header = 'L  chi    dchi    beta    dbeta    chi2_red',
               fmt = ['%.0f','%.3f','%.3f','%.5f','%.5f','%.2f',])
    pass
