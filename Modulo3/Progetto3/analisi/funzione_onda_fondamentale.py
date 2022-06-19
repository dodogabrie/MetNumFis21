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
    nlat = 10
    eta = 0.1
    data_dir = f'../dati/stato_fondamentale_singola/nlat{nlat}_eta{eta}/'
    files = [f for f in listdir(data_dir) if (isfile(join(data_dir, f)) and f.endswith('.dat'))]
    eta = 1e-2
    lattices = np.array([])
    for file in files:
        num = ''
        for m in file:
            if m.isdigit():
                num += m
        print(num)
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
#    ax.set_title(r"$\left|\psi_0\right|^2$ in funzione di $y$")

    ax.set_title("Ampiezza della funzione d'onda dello stato fondamentale")
    ax.set_ylabel(r'$\left|\psi_0(y)\right|^2$')
    ax.set_xlabel(r'$y$')

    ax.hist(lattices, bins = 80, density = True, label = 'dati', edgecolor = "white", linewidth = 0.1)

    xx = np.linspace(-5, 5, 1000)
    def f(x): 
        return np.exp(-x**2) * np.sqrt(1/(np.pi))
    ax.plot(xx, f(xx), label = 'teorica')
    plt.legend()
#    plt.savefig('../figure/stato_fondamentale_singola.png', dpi = 199)
    plt.show()

