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

def fit_gamma_nu(data_file):
    L, chi, dchi,_ ,_ , _ = np.loadtxt(data_file, unpack = True)
    remove = 0
    L = L[remove:]
    chi = chi[remove:]
    dchi = dchi[remove:]
    def fit_func(x, a, b, c):
        return a + b * x**c
    init = [-20, 0.17, 1.7,]
    xx = np.linspace(np.min(L), np.max(L), 1000)
    sigma = dchi
    w = 1/sigma**2
    pars, covm = curve_fit(fit_func, L, chi, init, sigma, absolute_sigma=False)
    chi2 =((w*(chi-fit_func(L,*pars))**2)).sum()
    ndof = len(L) - len(init)
    errors = np.sqrt(np.diag(covm))

    a, da = pars[0], errors[0]
    b, db = pars[1], errors[1]
    c, dc = pars[2], errors[2]

    print('Results:')
    print(f'a = {a} +- {da}')
    print(f'b = {b} +- {db}')
    print(f'c = {c} +- {dc}')
    print(f'chi red = {chi2/ndof}')

    plt.plot(xx, fit_func(xx, *pars))
    plt.errorbar(L, chi, yerr = dchi, fmt = '.')
    plt.show()
    x = L
    y = chi * L**(-c)
    dy = np.sqrt( (dchi * L**(-c))**2 + (dc * chi * np.log(L) * L**(-c) )**2 )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(x, y, yerr = dy, fmt = '.', color = 'k')
    ax.set_xlabel('L', fontsize = 14)
    ax.set_ylabel(r'$\chi_{max} \cdot L^{-\gamma/\nu}$', fontsize = 14)
    #ax.set_title(rf'Cumulante di Binder per $\beta$ = {beta}', fontsize = 15)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    ax.set_title(r'Plot di $\chi_{max} \cdot L^{-\gamma/\nu}$ al variare di $L$', fontsize=15)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_ylim(0.09, 0.13)
    ax.grid(alpha = 0.2)
    plt.tight_layout()
    plt.savefig('../figures/chiL_alla_gamma_su_nu.png', dpi = 200)
    plt.show()
    return

if __name__ == '__main__':
    data_file = '../data/fit_beta_chi_max.txt'
    fit_gamma_nu(data_file)
    pass
