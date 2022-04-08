"""
Verify FSS for chi, C, energy, |M| using beta_c from the fit and gamma/nu from
the theory.
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

def collapse():
    gamma_on_nu = 7/4
    beta_c = 0.44
    nu = 1
    beta = 1/8
    alpha = 0
    # Data to esplore divided by lateral size of grid
    nlats = [10, 20, 30, 40, 50, 60, 70, 80]

    # Quantity to estimate
    #estimate = 'chi'
    #estimate = 'ene'
    #estimate = 'magn'
    estimate = 'c'

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dict_title = {'chi': 'FSS per la suscettività', 'ene':'Energia', 'magn':'FSS per la magnetizzazione',
                  'c':'FSS per il calore specifico con correzione'}

    dict_name = {'chi': 'FSSchi.png', 'ene':'FSSene.png', 'magn':'FSSmagn.png',
                  'c':'FSSc_corr.png'}

    dict_y = {'chi': r'$\chi/L^{\gamma/\nu}$', 'ene':'$\epsilon$', 'magn': r'$|M|\cdot L^{1/8}$','c':'C'}
    dict_x = {'chi': r'$(\beta-\beta_c)\cdot L$', 'ene':'$\epsilon$', 'magn':r'$(\beta-\beta_c) \cdot L$','c':r'$(\beta -\beta_c)\cdot L$'}


    # Extract list of data files
    list_data = [f'../data/final_results/data_obs_nlat{n}_test_final.dat' for n in np.array(nlats)]
    # Dictionary for indices of searched quantities
    dict_val = dict(magn = [1, 2], ene = [3, 4], chi = [5, 6], c = [7, 8])

    i, di = dict_val[estimate] # Extract indices from dict

    # For every dataset (one for each nlat) plot the curve varing beta
    fig, ax = plt.subplots(figsize=(6, 6))
    color = iter(plt.cm.tab10(np.linspace(0, 1, len(nlats))))

    data = np.loadtxt(list_data[-1]) # Extract data
    y = data[:,i]
    max_corr = np.max(y)

    for d, l in zip(list_data, nlats):
        data = np.loadtxt(d) # Extract data
        x = data[:,0] # X coordinate is always beta
        y = data[:,i]
        rel_max = np.max(y)
        corr = (max_corr - rel_max)
        y = y + corr
        x = (x-beta_c)*l**(1/nu)
        yerr = data[:,di]
        ax.errorbar(x, y, yerr = yerr, label = f'L = {l}', fmt='.',
                    markersize = 5, c = next(color))

    #---- Belluire ----------------------------------------------------------------
    # Minor ticks and inner ticks
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=4)
    ax.tick_params('y', which='major', direction='in', length=4)
    ax.tick_params('y', which='minor', direction='in', length=2, left=True)
    ax.tick_params('x', which='minor', direction='in', length=2,bottom=True)

    # Ax font size
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.grid(alpha=0.3)
    ax.set_xlabel(dict_x[estimate], fontsize=12)
    ax.set_ylabel(dict_y[estimate], fontsize=12)
    ax.set_title(dict_title[estimate], fontsize = 13)
    plt.legend(fontsize=11)
#    plt.savefig(f'../figures/FSS/{dict_name[estimate]}', dpi=200)
    plt.show()
    return

if __name__ == '__main__':
    collapse()
    pass
