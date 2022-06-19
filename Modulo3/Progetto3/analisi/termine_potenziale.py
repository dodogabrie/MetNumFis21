import sys
sys.path.append('../../../utils/')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import json
import m1.readfile as rf
from m1.error import err_mean_corr

def extract_from_json(file):
    json_file = file[:-4] + '.json'
    with open(json_file, 'r') as f:
        datastore = json.load(f)
    return datastore

def get_data(file):
    data = rf.fastload(file)
    y2 = data[:,0]
    dy2 = data[:,1]
    return y2, dy2

def get_fit_data():
    N_per_eta = 3
    nlat_list = np.array([8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 23, 25, 30, 50, 70])
    eta_list = N_per_eta/nlat_list
    list_y2_mean = []
    list_y2_err = []
    for nlat, eta in zip(nlat_list, eta_list):
        filename = f'../dati/potential_term/data_eta{eta}_nlat{nlat}.dat'
        print('getting data from file:', filename)
        y2, _ = get_data(filename)
        list_y2_mean.append(np.mean(y2))
        _, dy2, _ = err_mean_corr(y2)
        list_y2_err.append(dy2)



    U = 1/2 + 1/(np.exp(N_per_eta) - 1)

    xx = np.linspace(0, np.max(eta_list), len(eta_list))
    yy = U * np.ones(len(eta_list))
    font = {'size': 12}
    plt.rc('font', **font)

    fig, ax = plt.subplots(1, 1)

    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\left<y^2\right>$')
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


    ax.errorbar(eta_list, list_y2_mean, yerr = list_y2_err, fmt = '.', label = 'Dati')
    ax.scatter(np.zeros(1), np.ones(1)*U, label = r'$\left<y^2\right>_c$ atteso', marker = 'x', s = 50, color = 'red')
    def f(x, a, b):
        return a + b*x**2

    x = eta_list
    y = np.array(list_y2_mean)
    dy = np.array(list_y2_err)
    sigma = np.array(dy)
    w = 1/sigma**2
    init = [1, 1]
    pars, covm = curve_fit(f, x, y, init, sigma, absolute_sigma=True)
    chi = ((w*(y-f(x,*pars))**2)).sum()
    ndof = len(y) - len(init)
    errors = np.sqrt(np.diag(covm))
    a, da = pars[0], errors[0]
    b, db = pars[1], errors[1]
    ax.plot(xx, f(xx, *pars), label = 'Fit')
    ax.legend()
    ax.set_title(r'Stima di $\left< y^2 \right>_c$ per $T = \hbar \omega /3$')
    plt.savefig('../figure/termine_potenziale/termine_potenziale.png', dpi = 200)
    plt.show()
    print('theoretical:', U)
    print('from fit:', a, '+-', da)
    print('b from fit:', b, '+-', db)
    print('chi2:', chi, 'ndof:', ndof, '... chi red:', chi/ndof)





if __name__ == '__main__':
    get_fit_data()


