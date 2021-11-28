"""
This file plot a MC history given the filename.
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################


import numpy as np
from m1.readfile import slowload, fastload
from os import listdir
from os.path import isfile, join
import time

def history_for_report(filename):
    """
    This function plots the firsts and the lasts points of a Montecarlo Hist.
    saved in the file 'filename'.
    """
    magn, ene = np.loadtxt(filename, unpack = True)
    import matplotlib.pyplot as plt
    fig , axs = plt.subplots(1, 2, sharey = True, figsize = (10,5))
    ax1, ax2 = axs
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.suptitle(r'L = 50, $\beta$ = 0.4326',)

    for ax in axs:
        ax.minorticks_on()
        ax.tick_params('x', which='major', direction='in', length=3)
        ax.tick_params('y', which='major', direction='in', length=3)
        ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
        ax.tick_params('x', which='minor', direction='in', length=1.5,bottom=True)


    ax1.plot(magn, linewidth = 0.7, color = 'brown')
    ax1.set_ylim(-1,1)
    ax1.set_xlim(0,1000)
    ax1.set_xlabel('Markov chain steps')
    ax1.set_ylabel('M')
    xticks = ax1.xaxis.get_major_ticks()
    xticks[-1].set_visible(False)

    ax2.plot(magn, linewidth = 0.7, color = 'brown')
    ax2.set_ylim(-1,1)
    ax2.set_xlim(99000,100000)
    ax2.set_xlabel('Markov chain steps')
    xticks = ax2.xaxis.get_major_ticks()
    xticks[0].set_visible(False)

#    plt.savefig('../figures/MC_history/historyMC_L50_043.png', dpi = 300)
    plt.show()

def single_history(filename):
    """
    This function plots a single history from file 'filename'.
    """
    data = fastload(filename.encode('UTF-8'), int(1e5))
    magn, ene = data.T
    import matplotlib.pyplot as plt
    plt.plot(ene)
    plt.show()

def history_varying_L(Ls, beta, N):
    """
    This function plots a set of hystory at given given values of L and a single
    value of beta.
    Parameters
    ----------
    Ls : list
        List containing the values of L to be evaluated.
    beta : float
        Value of beta to evaluated.
    N : int
        Number of data in the MC history.

    Returns
    -------
    """
    num_L = len(Ls)
    magn_list = np.empty((num_L, N))
    betas = []
    for i, L in enumerate(Ls):
        # Extract the closer beta to the given (from files) ###################
        data_dir = f'../data/nlat{L}/'
        closer_beta = _extract_closer_beta(data_dir, beta)
        #######################################################################
        filename = data_dir + 'data_beta' + str(closer_beta) + f'_nlat{L}.dat'
        data = fastload(filename.encode('UTF-8'), N)
        magn, ene = data.T
        magn_list[i] = magn
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(int(num_L/2), int(num_L/2), figsize = (14, 6))
    for i, ax, magn, l in zip(range(len(Ls)), axs.flat, magn_list, Ls):
        ax.scatter(np.arange(len(magn)), magn, s = 0.5, color = 'brown',
                   alpha = 0.4)
        ax.set_ylim(-1., 1.)
        ax.set_xlim(0, N)
        ax.set_title(f'L = {l}', fontsize = 12)
        if i % int(num_L/2) == 0:
            ax.set_ylabel('M', fontsize = 11 )
        if i >= int(num_L/2):
            ax.set_xlabel('MC step', fontsize = 11)
    plt.suptitle(rf'Storia MC di M per $\beta$ = {closer_beta:.4f}', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'../figures/history_simm_break_M/beta{closer_beta:.4f}.png', dpi = 300)
    plt.show()
    return

def _extract_closer_beta(data_dir, beta):
    """
    Extract closer beta values to 'beta' for a list of file in 'data_dir'.
    """
    # Extract files
    betas = []
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    for data_name in onlyfiles: # Extract all beta
        tmp_string = data_name.split('_nlat')[0] # beta comes first of '_nlat'
        beta_from_file = float(tmp_string.split('beta')[1])# beta comes after 'beta'
        betas.append(beta_from_file)
    betas = np.array(betas)
    closer_beta = betas[np.argmin(np.abs(betas - beta))]
    return closer_beta

def plot_histogram(L, N, bl, bm, bh):
    data_dir = f'../data/nlat{L}/'

    closer_beta = _extract_closer_beta(data_dir, bl)
    filename = data_dir + 'data_beta' + str(closer_beta) + f'_nlat{L}.dat'
    data = fastload(filename.encode('UTF-8'), N)
    magn_lower, _ = data.T

    closer_beta = _extract_closer_beta(data_dir, bm)
    filename = data_dir + 'data_beta' + str(closer_beta) + f'_nlat{L}.dat'
    data = fastload(filename.encode('UTF-8'), N)
    magn_medium, _ = data.T

    closer_beta = _extract_closer_beta(data_dir, bh)
    filename = data_dir + 'data_beta' + str(closer_beta) + f'_nlat{L}.dat'
    data = fastload(filename.encode('UTF-8'), N)
    magn_high, _ = data.T

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(4, 9))

    axs[0].hist(magn_lower, bins = 100, density = True, label = rf'$\beta$ = {bl}', color = 'brown')
    axs[0].set_xlim(-1, 1)
    axs[0].legend(loc = 'upper right', framealpha = 0.7)
    axs[0].set_ylabel('P(M)')

    axs[1].hist(magn_medium, bins = 100, density = True, label = rf'$\beta$ = {bm}', color = 'brown')
    axs[1].set_xlim(-1, 1)
    axs[1].legend(loc = 'upper right', framealpha = 0.7)
    axs[1].set_ylabel('P(M)')

    axs[2].hist(magn_high, bins = 100, density = True, label = rf'$\beta$ = {bh}', color = 'brown')
    axs[2].set_xlim(-1, 1)
    axs[2].legend(loc = 'upper right', framealpha = 0.7)
    axs[2].set_ylabel('P(M)')
    axs[2].set_xlabel('M')

    plt.suptitle(f'Distrib. di probabilità di M per L = {L}')
    plt.tight_layout()
    plt.savefig(f'../figures/P(M)_L{L}.png', dpi = 300)
    plt.show()
    return

if __name__ == '__main__':
    # Plots of first and last MC steps #######################################
#    filename = '../data/nlat20/data_beta0.4642480170287998_nlat20.dat'
#    history(filename)

    # A single MC history ####################################################
#    filename = '../data/nlat20/data_beta0.4642480170287998_nlat20.dat'
#    single_history(filename)

    # Plot of M history varying L ############################################
#    Ls = [20, 40, 60, 80]
#    beta = 0.436
#    N = int(1e5)
#    history_varying_L(Ls, beta, N)

    # Plot the hisogram of M for 2 values of beta
#    L = 50
#    N = int(1e5)
#    beta_low = 0.38
#    beta_medium = 0.45
#    beta_high = 0.48
    pass
#    plot_histogram(L, N, beta_low, beta_medium, beta_high)
