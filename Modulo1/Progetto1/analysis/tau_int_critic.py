"""
Compute Z factor of correlation near the critic points from a linear fit of
tau_int vs N lat.
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
#import matplotlib.pyplot as plt
from m1.readfile import fastload
from m1.error import err_mean_corr
from scipy.optimize import curve_fit
from scipy import stats
from plot_history import _extract_closer_beta

def compute_z(data_main_dir):
    beta_c = 0.4406
    L_fit, _, _, beta_c_fit, _, _ = np.loadtxt('../data/fit_beta_chi_max.txt', unpack = True)
    tau_c_list = []
    L_array = L_fit[:-1:2]
    beta_arr = beta_c_fit[:-1:2]
    for L, beta_c in zip(L_array, beta_arr):
        L = int(L)
        data_dir = data_main_dir + f'nlat{L}'
        beta_closer = _extract_closer_beta(data_dir, beta_c)
        file_name = data_dir + f'/data_beta{beta_closer}_nlat{L}.dat'
        data = fastload(file_name.encode('UTF-8'), int(1e5))
        magn, ene = data[:, 0], data[:, 1]
        tau_c, _, _ = err_mean_corr(magn)
        tau_c_list.append(tau_c)
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    tau_c_list = np.array(tau_c_list)
    tau_log = np.log(tau_c_list)
    L_log = np.log(L_array)
    regressor = LinearRegression().fit(L_log.reshape((len(L_log),1)), tau_log.reshape((len(tau_log),1)) )
    z = regressor.coef_[0][0]
    intercept = regressor.intercept_[0]
    xx = np.linspace(np.min(L_log), np.max(L_log), 500)

    fig, ax = plt.subplots()
    ax.plot(xx, z*xx + intercept, c = 'brown', label = f'z = {z:.2f}')
    ax.scatter(L_log, tau_log, c = 'k', s = 17)
    ax.set_ylabel(r'$\log(\tau_c)$', fontsize = 14)
    ax.set_xlabel(r'$\log(L)$', fontsize = 14)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=4)
    ax.tick_params('y', which='major', direction='in', length=4)
    ax.tick_params('y', which='minor', direction='in', length=2, left=True)
    ax.tick_params('x', which='minor', direction='in', length=2,bottom=True)

    # Ax font size
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(r"$\log(\tau_c)$ vs $\log(L)$ al punto critico", fontsize = 15)

    ax.grid(alpha=0.3)
    ax.legend(fontsize = 13, loc = 'lower right')
    plt.savefig('../figures/tau_c_vs_L_interpolazione.png', dpi = 200)
    plt.show()
    return

if __name__ == '__main__':
    data_main_dir = '../data/'
    compute_z(data_main_dir)
