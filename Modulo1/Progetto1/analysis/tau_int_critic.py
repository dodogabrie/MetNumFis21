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
    tau_c_list = np.array(tau_c_list)
    plt.scatter(L_array, tau_c_list)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    return

if __name__ == '__main__':
    data_main_dir = '../data/'
    compute_z(data_main_dir)
