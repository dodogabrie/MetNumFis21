### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

from m1.error import err_mean_corr, err_naive, bootstrap_corr
from m1.readfile import fastload
from estimator import compute_chi, compute_c
import numpy as np
import time 
from numba import njit

def test():
    L = 10
    beta = 0.3
    dir_data = b"data/data1.dat"
    start = time.time()
    data = fastload(dir_data, int(1e6))
    # Processing M and E
    magn, ene = data[:, 0], data[:, 1]
    m_abs = np.abs(magn)
    start = time.time()
    tau_magn, err_magn, Ck_magn = err_mean_corr(m_abs, kmax=int(1e2))
    tau_ene, err_ene, Ck_ene = err_mean_corr(ene, kmax=int(1e2))
    print(f'tau_magn = {tau_magn} (or as lenght of Ck: {len(Ck_magn)})')
    print(f'tau_magn = {tau_ene} (or as lenght of Ck: {len(Ck_ene)})')
    print(f'magn = {np.mean(m_abs)} +- {err_magn}')
    # Processing chi and c
    chi = compute_chi(m_abs, (L, beta))
    c = compute_c(ene, (L, beta))
    d_chi = bootstrap_corr(m_abs, 2000, compute_chi, param = (L, beta))
    d_c = bootstrap_corr(ene, 2000, compute_c, param = (L, beta))
    print(chi, '+-', d_chi)
    print(c, '+-', d_c)

    import matplotlib.pyplot as plt
    plt.plot(Ck_magn)
    plt.show()
    return 

if __name__ == '__main__':
    test()
