import sys
sys.path.append('../../../utils/')
import numpy as np
import json
from numba import njit
import m1.readfile as rf
from m1.error import err_mean_corr, bootstrap_corr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

def extract_from_json(json_file):
    print(json_file)
    with open(json_file, 'r') as f:
        datastore = json.load(f)
    return datastore

def get_data(file):
    file = file[:-5]
    data_obs1 = rf.fastload(file + '1.dat')
    data_obs2 = rf.fastload(file + '2.dat')
    k_list = data_obs1[0, :]
    data_obs1 = data_obs1[1:, :]
    data_obs2 = data_obs2[1:, :]
    return k_list, data_obs1, data_obs2

def fit_data(x, y, dy, init = [0, 0.5, 1], plot_results = False):
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
#    print('chi_red = ', chi/ndof)
#    print(Delta_E, dDelta_E)
    if plot_results:
       plt.errorbar(x, y, yerr = dy, fmt = '.')
       xx = np.linspace(np.min(x), np.max(x), 1000)
       plt.plot(xx, f(xx, *pars))
       plt.show()
    return A, dA, Delta_E, dDelta_E
    

if __name__ == '__main__':
    data_dir = '../dati/lim_continuum_E1_E0/'
    onlyfiles = [f for f in listdir(data_dir) if (isfile(join(data_dir, f)) and f.endswith('1.dat') and 'lattice' not in f)] # extract list of data file
    jsonfiles = [f for f in listdir(data_dir) if (isfile(join(data_dir, f)) and f.endswith('.json') and 'lattice' not in f)] # extract list of data file
    print(onlyfiles)
    list_DE = []
    list_dDE = []
    eta_list = []
    for file_name, json_file in zip(onlyfiles, jsonfiles):
        data_dict = extract_from_json(data_dir + json_file)
        eta = data_dict["eta"]
        k_list, obs1, obs2 = get_data(data_dir + file_name)
        print(file_name, eta)
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
        _, _, Delta_E, dDelta_E = fit_data(x, y1, dy1)
        #fit_data(x, y2, dy2)
        list_DE.append(Delta_E)
        list_dDE.append(dDelta_E)
        eta_list.append(eta)
    list_DE = np.array(list_DE)
    list_dDE = np.array(list_dDE)
    eta_list = np.array(eta_list)
    plt.scatter(eta_list**2, list_DE)
    plt.show()

