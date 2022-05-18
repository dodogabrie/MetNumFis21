import sys
sys.path.append('../../../utils/')
import numpy as np
from os import listdir 
from os.path import isfile, join
import json
from numba import njit
import m1.readfile as rf
from m1.error import err_mean_corr, bootstrap_corr

if __name__ == '__main__':
    # read data
    data_dir = '../dati/stato_fondamentale/'
    files = [f for f in listdir(data_dir) if (isfile(join(data_dir, f)) and f.endswith('.dat'))]
    eta = 1e-2
    lattices = np.array([])
    for file in files:
        print(file)
        lattice = np.loadtxt(data_dir + file)
        lattices = np.append(lattices, lattice)

    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots(1, 1)

    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    ax.set_xlabel('y')
    ax.set_ylabel('p(y)')
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax.hist(lattices, bins = 100, density = True)
    plt.show()

