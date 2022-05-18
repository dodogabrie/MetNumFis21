import sys
sys.path.append('../../../utils/')
import numpy as np
import json
from numba import njit
import m1.readfile as rf
from m1.error import err_mean_corr, bootstrap_corr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def extract_from_json(file):
    json_file = file + '.json'
    with open(json_file, 'r') as f:
        datastore = json.load(f)
    return datastore


def get_data(file):
    data_obs1 = rf.fastload(file + '_obs1.dat')
    data_obs2 = rf.fastload(file + '_obs2.dat')
    k_list = data_obs1[0, :]
    data_obs1 = data_obs1[1:, :]
    data_obs2 = data_obs2[1:, :]
    return k_list, data_obs1, data_obs2

def fit_data(x, y, dy, init = [0, 0.5, 1]):
    def f(x, bias, A, Delta_E):
        return bias + A * np.exp(-Delta_E * x)
    # perché bias?
    # Perché, non essendo a temperatura nulla, si introduce un bias sulla quantità connessa.
    sigma = dy
    w = 1/sigma**2
    pars, covm = curve_fit(f, x, y, init, sigma, absolute_sigma=False)
    chi = ((w*(y-f(x,*pars))**2)).sum()
    ndof = len(y) - len(init)
    errors = np.sqrt(np.diag(covm))
    bias, dbias = pars[0], errors[0]
    A, dA = pars[1], errors[1]
    Delta_E, dDelta_E = pars[2], errors[2]
    print('chi_red = ', chi/ndof)
    print(Delta_E, dDelta_E)
    plt.errorbar(x, y, yerr = dy, fmt = '.')
    xx = np.linspace(np.min(x), np.max(x), 1000)
    plt.plot(xx, f(xx, *pars))
    plt.show()
    return A, dA, Delta_E, dDelta_E
    

if __name__ == '__main__':
    nlat = 500
    eta = 0.8
    data_dir = '../dati/gap_energy/'
    file_name = f'{eta}_{nlat}_Gap_energy'
    k_list, obs1, obs2 = get_data(data_dir + file_name)
    
    x = eta * k_list
    y1 = np.mean(obs1, axis=0) # array con una y per ogni k nella k_list
    y2 = np.mean(obs2, axis=0) # array con una y per ogni k nella k_list
    dy1 = np.empty(y1.shape)
    dy2 = np.empty(y2.shape)
    for i, y in enumerate(obs1.T): # Va calcolato l'errore per ogni k nella k_list
        _, err, _ = err_mean_corr(y)
        dy1[i] = err
    for i, y in enumerate(obs2.T): # Va calcolato l'errore per ogni k nella k_list
        _, err, _ = err_mean_corr(y)
        dy2[i] = err

    for yy, dyy in zip(y1, dy1):
        print(yy, dyy)
    fit_data(x, y1, dy1)
    fit_data(x, y2, dy2)
