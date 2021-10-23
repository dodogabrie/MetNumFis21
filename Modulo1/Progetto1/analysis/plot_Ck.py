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
import matplotlib.pyplot as plt
import time

def Ck_plot(list_file):

    for filename in list_file:
        # Extract beta from file name
        tmp_string = filename.split('_nlat')[0] # beta comes first of '_nlat'
        beta = float(tmp_string.split('beta')[1])# beta comes after 'beta'

        magn, ene = np.loadtxt(filename, unpack = True)
        tau, error, Ck = err_mean_corr(magn, 400)
        plt.plot(Ck, linewidth = 1, label = rf'$\beta$ = {beta:.4f}')

    plt.title(rf'L = 80')
    plt.minorticks_on()
    plt.tick_params('x', which='major', direction='in', length=3)
    plt.tick_params('y', which='major', direction='in', length=3)
    plt.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    plt.tick_params('x', which='minor', direction='in', length=1.5,bottom=True)

    plt.xlabel('k')
    plt.ylabel('C(k)')
    plt.legend()

    #plt.savefig('../figures/Ck_L80.png', dpi = 300)
    plt.show()

if __name__ == '__main__':
    L = 50
    list_beta = [0.38, 0.4009030737312974, 0.4200588742480439, 0.4345577217609058,0.4405432022366073,
                 0.4505067030352683, 0.4684809360062864, 0.47091621903968484, 0.48]
    list_file = []
    for beta in list_beta:
        list_file.append(f'../data/nlat{L}/data_beta{beta}_nlat{L}.dat')
    Ck_plot(list_file)
