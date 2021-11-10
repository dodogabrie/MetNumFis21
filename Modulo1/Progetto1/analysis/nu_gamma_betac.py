"""
Fit to estimate gamma on nu given L (x) and the relative chi_max (y)
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

def fit_nu_gamma_betac(data_file):
    L, _, _, beta_max ,dbeta_max , _ = np.loadtxt(data_file, unpack = True)
    remove = 0
    L = L[remove:]
    beta_max = beta_max[remove:]
    dbeta_max = dbeta_max[remove:]
    def fit_func(x, betac, xbar, nu):
        return betac + xbar * x**(-1/nu)
    init = [0.43, -0.5, 0.93,]
    xx = np.linspace(np.min(L), np.max(L), 1000)
    sigma = dbeta_max
    w = 1/sigma**2
    pars, covm = curve_fit(fit_func, L, beta_max, init, sigma, absolute_sigma=False)
    chi2 =((w*(beta_max-fit_func(L,*pars))**2)).sum()
    ndof = len(L) - len(init)
    errors = np.sqrt(np.diag(covm))

    betac, dbetac = pars[0], errors[0]
    xbar, dxbar = pars[1], errors[1]
    nu, dnu = pars[2], errors[2]

    print('Results:')
    print(f'beta_c = {betac} +- {dbetac}')
    print(f'xbar = {xbar} +- {dxbar}')
    print(f'nu = {nu} +- {dnu}')
    print(f'chi red = {chi2/ndof}')

    plt.plot(xx, fit_func(xx, *pars))
    plt.errorbar(L, beta_max, yerr = dbeta_max, fmt = '.')
    plt.show()

#    fig, ax = plt.subplots(figsize=(6, 5))
#    ax.errorbar(x, y, yerr = dy, fmt = '.', color = 'k')
#    ax.set_xlabel('L', fontsize = 14)
#    ax.set_ylabel(r'$\chi_{max} \cdot L^{-\gamma/\nu}$', fontsize = 14)
#    #ax.set_title(rf'Cumulante di Binder per $\beta$ = {beta}', fontsize = 15)
#    ax.minorticks_on()
#    ax.tick_params('x', which='major', direction='in', length=3)
#    ax.tick_params('y', which='major', direction='in', length=3)
#    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
#    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
#    ax.set_title(r'Plot di $\chi_{max} \cdot L^{-\gamma/\nu}$ al variare di $L$', fontsize=15)
#    ax.tick_params(axis='both', labelsize=11)
#    ax.set_ylim(0.09, 0.13)
#    ax.grid(alpha = 0.2)
#    plt.tight_layout()
#    plt.savefig('../figures/chiL_alla_gamma_su_nu.png', dpi = 200)
#    plt.show()
    return

if __name__ == '__main__':
    data_file = '../data/fit_beta_chi_max.txt'
    fit_nu_gamma_betac(data_file)
    pass
