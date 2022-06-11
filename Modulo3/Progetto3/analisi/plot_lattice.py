import sys
sys.path.append('../../../utils/')
import numpy as np
import matplotlib.pyplot as plt
import json
from numba import njit
import m1.readfile as rf
from m1.error import err_mean_corr, bootstrap_corr
from scipy.optimize import curve_fit

def plot_lattice(filename, x, label, ax = None):
    data = np.loadtxt(filename)
    if ax == None:
        plt.plot(x, data, label = label)
    else: 
        ax.plot(x, data, label = label)
    return ax

def plot_all():
    data_dir = "../dati/plot_cammini/"
    eta = 0.001
    nlats = [100, 200, 1000]
    filenames = [f'data_eta{eta}_nlat{nlat}.dat' for nlat in nlats] 
    fig, ax = plt.subplots(1, 1)
    for filename, nlat in zip(filenames, nlats):
        x = np.linspace(0, eta*(nlat-1), nlat)
        if x[1]-x[0] != eta:
            raise ValueError('x non è uniforme')
        ax = plot_lattice(data_dir + filename, x, label = f'N = {nlat}', ax = ax)

    ax.set_title('Forma dei cammini al variare di N')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('y')

    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
 
    ax.grid(alpha=0.3)
    plt.legend()
    plt.savefig('../figure/lattice/lattice_vary_nlat.png', dpi = 150)
    plt.show()
    plt.show()

def check_termalization_nlat():
    data_dir = "../dati/plot_cammini/"
    eta = 0.001
    nlats = [1000, 5000, 7000]
    filenames = [f'data_eta{eta}_nlat{nlat}.dat' for nlat in nlats] 
    fig, ax = plt.subplots(1, 1)
    for filename, nlat in zip(filenames, nlats):
        data = rf.fastload(data_dir + filename)
        y2 = data[:,0]
        dy2 = data[:,1]
        ax.plot(y2, label = f'{nlat}')

#    ax.set_title('Tempo di termalizzazione al variare di N')
    ax.set_xlabel('Tempo MC')
    ax.set_ylabel(r'$\left<y^2\right>$')
    ax.set_xlim(-1e4, 5e5)
    ax.set_ylim(0, .5)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
 
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('../figure/termalizzazione/termalization_y2_vary_nlat.png', dpi = 150)
    plt.show()
    return

def check_termalization_eta():
    data_dir = "../dati/plot_cammini/vary_eta/"
    nlat = 300
    etas = [1e-1, 1e-2, 1e-3]
    filenames = [f'data_eta{eta}_nlat{nlat}.dat' for eta in etas] 
    fig, ax = plt.subplots(1, 1)
    for filename, eta in zip(filenames, etas):
        data = rf.fastload(data_dir + filename)
        y2 = data[:,0]
        dy2 = data[:,1]
        ax.plot(y2, label = f'{eta}')

#    ax.set_title('Tempo di termalizzazione al variare di N')
    ax.set_xlabel('Tempo MC')
    ax.set_ylabel(r'$\left<y^2\right>$')
#    ax.set_xlim(0, 2e5)
#    ax.set_ylim(0, .5)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
 
    ax.legend()
    ax.grid(alpha=0.3)
#    plt.savefig('../figure/termalizzazione/termalization_y2_vary_nlat.png', dpi = 150)
    plt.show()
    return


if __name__ == "__main__":
#    check_termalization_nlat()
    plot_all()
