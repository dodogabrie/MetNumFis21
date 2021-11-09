"""
This file plots the lattice after some update with a metropolis algorithm.
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################


import numpy as np
from m1.error import err_mean_corr, err_naive, bootstrap_corr
import ising
from os import listdir
from os.path import isfile, join
import time
import matplotlib.pyplot as plt

def plot_simulations(beta_array, param, simulate = False, save_fig = False):
    fig = plt.figure(figsize = (20, 6))
    rowcolnum = 131
    for beta in beta_array:
        if simulate:
            ising.do_calc(*param, beta, save_data = False, save_lattice = True)
        ax = plot_lattice(f'../data/lattice_matrix/lattice_nlat{param[0]}_beta{beta}', fig = fig, rowcolnum =rowcolnum)
        ax.set_title(rf'$\beta$ = {beta}', fontsize = 15)
        rowcolnum += 1
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'../figures/plot3D_lattice.png', dpi = 200)
    plt.show()
    return

def plot_lattice(lattice_file, fig = None, rowcolnum = None):
    lattice = np.loadtxt(lattice_file)
    if fig == None: fig = plt.figure()
    ax = fig.add_subplot(rowcolnum,projection='3d')
    x = np.arange(len(lattice))
    y = np.arange(len(lattice))
    X, Y = np.meshgrid(x, y)
    ax.scatter(X, Y, lattice, s = 4)
    ax.set_xlabel('x', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_zlabel('spin', rotation=90, fontsize = 15)
    ax.set_zticks([-1,1])
    ax.tick_params(axis='z', labelsize=12)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    return ax

if __name__ == '__main__':
    iflag = 1 # Start hot or cold
    i_decorrel = 100 # Number of decorrelation for metro
    extfield = 0. # External field
    measures = int(1e3) # Number of measures
    L = 70
    param = [L, iflag, measures, i_decorrel, extfield]
    beta_array = [0.38, 0.44, 0.48]
    plot_simulations(beta_array, param)
    pass
