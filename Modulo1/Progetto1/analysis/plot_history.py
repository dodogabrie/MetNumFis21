"""
This file plot a MC history given the filename
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################


import numpy as np
from m1.readfile import slowload, fastload
import time

def history_for_report(filename):
    magn, ene = np.loadtxt(filename, unpack = True)
    import matplotlib.pyplot as plt
    fig , axs = plt.subplots(1, 2, sharey = True, figsize = (10,5))
    ax1, ax2 = axs
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.suptitle(r'L = 50, $\beta$ = 0.4326',)

    for ax in axs:
        #ax.grid(b=True, color='grey', linestyle='-', alpha=0.3)
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
    data = fastload(filename.encode('UTF-8'), int(1e5))
    magn, ene = data.T
    import matplotlib.pyplot as plt
    plt.plot(ene)
    plt.show()
    

if __name__ == '__main__':
    filename = '../data/nlat20/data_beta0.4642480170287998_nlat20.dat'
#    history(filename)
    single_history(filename)
