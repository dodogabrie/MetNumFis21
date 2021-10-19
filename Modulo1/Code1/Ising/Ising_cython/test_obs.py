### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

from m1.error import err_mean_corr, err_naive, bootstrap_corr
from m1.readfile import slowload, fastload
from estimator import compute_chi, compute_c
import numpy as np
import time 
from numba import njit
from os import listdir
from os.path import isfile, join

from joblib import Parallel, delayed

def test():
    nlat = 20
    n_jobs = 8
    M = 2000 #sicuri?
    data_dir = "data/nlat20/"
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
   
    def extract_obs(data_name):
        # Extract beta from filename 
        file_test = data_name
        tmp_string = file_test.split('_nlat')[0]
        beta = float(tmp_string.split('beta')[1])
 
        data = fastload(data_name.encode('UTF-8'), int(1e5))
        magn, ene = data[:, 0], data[:, 1]
        m_abs = np.abs(magn)
        chi = compute_chi(magn, (nlat, beta))
        c = compute_c(ene, (nlat, beta))
        dchi = bootstrap_corr(magn, M, compute_chi, param = (nlat, beta))
        dc = bootstrap_corr(ene  , M, compute_c  , param = (nlat, beta))
        m_abs_mean = np.mean(m_abs)
        dm_abs = err_naive(m_abs)
        ene_mean = np.mean(ene)
        dene = err_naive(ene)
        return [beta, m_abs_mean, dm_abs, ene_mean, dene, chi, dchi, c, dc]

    list_outputs = [extract_obs(data_dir + data_name) for data_name in onlyfiles]

    data = np.array(list_outputs)

    import matplotlib.pyplot as plt
    plt.scatter(data[:,0], data[:, 1], label = 'magn')
    plt.legend()
    plt.show()
    return 

if __name__ == '__main__':
    test()
