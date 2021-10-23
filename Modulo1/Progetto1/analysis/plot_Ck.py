"""
This file plot Ck of M given beta's values
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################


import numpy as np
from m1.error import err_mean_corr, err_naive, bootstrap_corr
from m1.readfile import slowload, fastload
from os import listdir
from os.path import isfile, join
import time

def Ck_plot(list_file, data_dir, M):
    # Extract all file names in the folder
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    nfile = len(onlyfiles)
    M_arr = np.arange(M-1)
    beta_arr = np.empty(nfile)
    Ck_matrix = np.empty((nfile, M-1))
    for i, data_name in enumerate(onlyfiles):
        print(f'{i} su {nfile}', end='\r')
        # Extract beta from file name 
        tmp_string = data_name.split('_nlat')[0] # beta comes first of '_nlat'
        beta = float(tmp_string.split('beta')[1])# beta comes after 'beta'
        beta_arr[i] = beta 
        # Load all the magnetization and energy at fixed beta
        data = fastload((data_dir + data_name).encode('UTF-8'), int(1e5))
        magn, ene = data[:, 0], data[:, 1]
        tau, error, Ck = err_mean_corr(magn, M)
        Ck_matrix[i] = Ck 
    beta_sort = np.argsort(beta_arr)
    beta_arr = beta_arr[beta_sort]
    Ck_matrix = Ck_matrix[beta_sort]

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import ScalarMappable

    fig = plt.figure(figsize = (15, 5))
    fonts = {'size' : 14}
    plt.rc('font', **fonts)
    plt.suptitle(r'Autocorrelazione di M al variare di $\beta$')
    gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    X, Y = np.meshgrid(M_arr, beta_arr)

    qcs = ax1.contourf(Y,X, Ck_matrix, 20, cmap = 'BuGn')
    fig.colorbar(ScalarMappable(norm=qcs.norm, cmap=qcs.cmap), ax = ax1, pad = 0.02,)
    ax1.set_xlabel(r'$\beta$')
    ax1.set_ylabel('k')
    ax1.set_xlim(0.42,0.455)
 
    for filename in list_file:
        # Extract beta from file name
        tmp_string = filename.split('_nlat')[0] # beta comes first of '_nlat'
        beta = float(tmp_string.split('beta')[1])# beta comes after 'beta'

        magn, ene = np.loadtxt(filename, unpack = True)
        tau, error, Ck = err_mean_corr(magn, M)
        plt.plot(Ck, linewidth = 1, label = rf'$\beta$ = {beta:.4f}')

    ax2.set_title(rf'L = 80')
    ax2.minorticks_on()
    ax2.tick_params('x', which='major', direction='in', length=3)
    ax2.tick_params('y', which='major', direction='in', length=3)
    ax2.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax2.tick_params('x', which='minor', direction='in', length=1.5,bottom=True)

    ax2.set_xlabel('k')
    ax2.set_ylabel('C(k)')
    ax2.legend(fontsize = 10)

    #plt.savefig('../figures/Ck_L80.png', dpi = 300)
    plt.show()

if __name__ == '__main__':

    #list_file.append(f'../data/nlat{L}/data_beta{beta}_nlat{L}.dat')
    M = 9000
    L = 80
    list_beta = [0.38, 0.4302356355217024, 0.4477279361553648, 0.48]
    list_file = []
    for beta in list_beta:
        list_file.append(f'../data/nlat{L}/data_beta{beta}_nlat{L}.dat')

#    M = 10000
    L = 80
    data_dir = f'../data/nlat{L}/'
    Ck_plot(list_file, data_dir, M)
