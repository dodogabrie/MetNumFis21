import sys
import os
import time
sys.path.append('../../../utils/')
sys.path.append('./')
import numpy as np
from os import listdir 
from os.path import isfile, join
import json
from numba import njit
import m1.readfile as rf
from m1.error import err_mean_corr, bootstrap_corr
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.integrate import quad
from my_hermite import psi2


# Somma dei polinomi di Hermite
def sum_stati(x, N, eta, num_stati = 10):
    T = 1/(N*eta)
#    list_stati = [stato_fondamentale, primo_eccitato, secondo_eccitato]
    list_energie = 1/2 + np.arange(0, num_stati)
#    list_energie = list_energie[:num_stati]
    tot = 0
    for i, E in enumerate(list_energie):
        tot += psi2(i, x)*np.exp(-E/T)
    return tot

def normalizer(f, x, args = ()):
    print(*args)
    return f(x, *args)/quad(f, -np.inf, np.inf, args = args)[0]
    
def plot_histogram(eta, nlat, ax = None, plot_teo = False, teorica = None, **kwargs):
    T = 1/(nlat*eta)
    print('T = ', T)
    print('U = ', 1/2 + 1/(np.exp(1/T) - 1))

    data_dir = f'../dati/stato_fondamentale_singola/nlat{nlat}_eta{eta}/'
    files = [f for f in listdir(data_dir) if (isfile(join(data_dir, f)) and f.endswith('.dat'))]
#    lattices = np.array([])
    list_num = np.empty(len(files))
    max_val = -np.infty
    min_val = np.infty
    # Pre-evaluation of max, min and order of files
    t = time.time()
    lattice = np.loadtxt(data_dir + files[0])
    t_numpy = time.time() - t
    t = time.time()
    lattice = pd.read_csv(data_dir + files[0], sep=" ", header=None).to_numpy()
    t_pandas = time.time() - t
    if t_numpy > t_pandas:
        numpy_method = False
    else:
        numpy_method = True
    count_data = 0
    for i, file in enumerate(files):
        num = ''
        for m in file:
            if m.isdigit():
                num += m
        list_num[i] = num
        if numpy_method:
            lattice = np.loadtxt(data_dir + file)
        else:
            lattice = pd.read_csv(data_dir + file, sep=" ", header=None).to_numpy()
        M = np.max(lattice)
        m = np.min(lattice)
        if m < min_val: min_val = m
        if M > max_val: max_val = M
        count_data += len(lattice)
    print('number of data: ', count_data)
#        os.remove(data_dir+file)
    # Create histogram
    lim = max(np.abs(min_val), np.abs(max_val))
    n_bins = 200
    bins = np.linspace(-lim, lim, n_bins)
    myhist = np.zeros(n_bins-1)
    htemp, jnk = np.histogram(np.array([]), bins)

    sort_num = np.argsort(list_num)
    files = np.array(files)[sort_num]
    list_num = list_num[sort_num]
    for file, n in zip(files, list_num):
        print(n, end = '\r')
        if numpy_method:
            lattice = np.loadtxt(data_dir + file)
        else:
            lattice = pd.read_csv(data_dir + file, sep=" ", header=None).to_numpy()
#        lattice = np.loadtxt(data_dir + file)
        htemp, jnk = np.histogram(lattice, bins)
        myhist += htemp
#        lattices = np.append(lattices, lattice)
    print('maximum found:', max_val)
    print('minimum found:', min_val)

    if ax == None:
        fig, ax = plt.subplots(1, 1)

    if plot_teo:
        xx = np.linspace(-5, 5, 1000)
        f = teorica 
        print(nlat, eta)
        label = ''
        if nlat == 1000: label = r'$P(N = 1000)$'
        if nlat == 10: label = r'$P(N = 15)$'
        if nlat == 5: label = r'$P(N = 5)$'
        ax.plot(xx, normalizer(f, xx, args = (nlat, eta)), label = label)

    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    ax.set_xlabel('y')
    ax.set_ylabel('p(y)')
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
#    ax.set_title(r"$\left|\psi_0\right|^2$ in funzione di $y$")

    ax.set_title(r"Istogramma dei valori di $y$ e curve attese")
    ax.set_ylabel(r'$P(y)$')
    ax.set_xlabel(r'$y$')

#    ax.hist(lattices, bins = 80, density = True, label = 'dati', edgecolor = "white", linewidth = 0.1, alpha = 0.5)
    norm_hist = myhist/(np.sum(myhist) * np.diff(bins)) 
    ax.stairs(norm_hist, jnk, fill = True, **kwargs)#label = 'stato fondamentale', color = 'black')
#    ax.vlines(jnk, 0, norm_hist.max(), colors='w', lw = 0.1)


if __name__ == '__main__':
    # read data
    fig, ax = plt.subplots(1, 1)
#    nlat_list = [5, 10, 20, 1000]
    nlat_list = [1000, 10, 5]
    eta_list = [0.1]*len(nlat_list)
    func_list = [sum_stati]*len(nlat_list)
#    eta_list = [0.1, 0.01]#, 0.001]
#    nlat_list = [100]*len(eta_list)
    plot_teo = True
    for nlat, eta, teorica in zip(nlat_list, eta_list, func_list):
        label = f'Dati: N = {nlat}'
#        label = f'eta = {eta}'
        plot_histogram(eta, nlat, ax, plot_teo = plot_teo, teorica = teorica, alpha = 0.4, label = label)
    plt.legend()
    plt.savefig('../figure/stato_fondamentale_singola_somma.png', dpi = 200)
    plt.show()

