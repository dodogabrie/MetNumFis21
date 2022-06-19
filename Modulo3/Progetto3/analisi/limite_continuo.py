import sys
sys.path.append('../../../utils/')
import numpy as np
import json
import os
from numba import njit
import m1.readfile as rf
from m1.error import err_mean_corr, bootstrap_corr
from scipy.optimize import curve_fit
from os import listdir
from os.path import isfile, join
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def extract_from_json(json_file):
    with open(json_file, 'r') as f:
        datastore = json.load(f)
    return datastore

def get_data(file):
    file = file[:-5]
    data_obs1 = pd.read_csv(file + '1.dat', sep=" ", header=None).to_numpy()
    #rf.fastload(file + '1.dat')
    data_obs2 = pd.read_csv(file + '2.dat', sep=" ", header=None).to_numpy()
    k_list = data_obs1[0, :]
    data_obs1 = data_obs1[1:, :]
    data_obs2 = data_obs2[1:, :]
    return k_list, data_obs1, data_obs2

def fit_data(x, y, dy, init = [0, 0.5, 1], eta = 0, ax = None, c = 'k', **kwargs):
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
    if ax is not None:
       ax.errorbar(x, y, yerr = dy, fmt = '.', label = f'eta = {eta}', c = c)
       xx = np.linspace(np.min(x), np.max(x), 1000)
       ax.plot(xx, f(xx, *pars), c = c, **kwargs)
    return A, dA, Delta_E, dDelta_E, ax


def elaborate_all(folder_parent = 'beta_omega_20', kind = ''):
    if kind == 'sconnessa':
        kind = '_solo_sconnessa'
        kind_1 = '_sconnessa'
    else:
        kind_1 = ''
    data_dir = '../dati/lim_continuum_E1_E0' + kind + '/'+ folder_parent + '/'
    onlyfiles = [f for f in listdir(data_dir) 
            if (isfile(join(data_dir, f)) and f.endswith('1.dat') and 'lattice' not in f)] # extract list of data file
    jsonfiles = [f for f in listdir(data_dir) 
            if (isfile(join(data_dir, f)) and f.endswith('.json') and 'lattice' not in f)] # extract list of data file
    list_DE1 = []
    list_dDE1 = []
    list_DE2 = []
    list_dDE2 = []
    eta_list = []
    if not os.path.exists(os.path.dirname(data_dir + 'elaborated/')): # If the directory does not exist
        os.makedirs(os.path.dirname(data_dir + 'elaborated/'))
        print('created folder for the data elaborated...')
    for file_name in onlyfiles:
        print('Elaboro:', file_name)
        identificative = file_name.split('_Gap_energy_')[0]
        json_file = [f for f in jsonfiles if identificative in f][0]
#        fig, ax = plt.subplots(1,1)
        data_dict = extract_from_json(data_dir + json_file)
        eta = data_dict["eta"]
        nlat = data_dict["nlat"]
        if nlat <=15: continue # Ricomincia da capo con il prossimo file
        k_list, obs1, obs2 = get_data(data_dir + file_name)
        print(file_name, eta)
        x = k_list
#        x = eta * k_list
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
        _, _, Delta_E1, dDelta_E1, _ = fit_data(x, y1, dy1,)# ax = ax)
        Delta_E1 /= eta
        dDelta_E1 /= eta
        new_file = file_name[:-5] + '1_elaborato.dat'
        np.savetxt(data_dir + 'elaborated/' + new_file, np.c_[x, y1, dy1])

        _, _, Delta_E2, dDelta_E2, _ = fit_data(x, y2, dy2,)# ax = ax)
        Delta_E2 /= eta
        dDelta_E2 /= eta
        new_file = file_name[:-5] + '2_elaborato.dat'
        np.savetxt(data_dir + 'elaborated/' + new_file, np.c_[x, y1, dy1])

        list_DE1.append(Delta_E1)
        list_dDE1.append(dDelta_E1)
        list_DE2.append(Delta_E2)
        list_dDE2.append(dDelta_E2)
        eta_list.append(eta)
#        plt.show()
    list_DE1 = np.array(list_DE1)
    list_dDE1 = np.array(list_dDE1)
    list_DE2 = np.array(list_DE2)
    list_dDE2 = np.array(list_dDE2)
    eta_list = np.array(eta_list)
    np.savetxt(data_dir + 'limite_continuo_1' + kind_1 + '.dat', np.column_stack((eta_list**2, list_DE1, list_dDE2)))
    np.savetxt(data_dir + 'limite_continuo_2' + kind_1 + '.dat', np.column_stack((eta_list**2, list_DE2, list_dDE2)))
    plt.scatter(eta_list**2, list_DE1)
    plt.show()
    return

def plot_exponential(folder_parent = 'beta_omega_20', gap = 1, kind = ''):
    if kind == 'sconnessa':
        kind = '_solo_sconnessa'
    data_dir = '../dati/lim_continuum_E1_E0' + kind + '/' + folder_parent + '/elaborated/'
    json_dir = '../dati/lim_continuum_E1_E0' + kind  + '/' + folder_parent + '/' + '/'
    files = [f for f in listdir(data_dir)]
    onlyfiles = [f for f in listdir(data_dir) 
                 if (isfile(join(data_dir, f)) and f'obs{gap}' in f and 'lattice' not in f)] # extract list of data file
    jsonfiles = [f for f in listdir(json_dir) 
                 if (isfile(join(json_dir, f)) and f.endswith('.json') and 'lattice' not in f)] # extract list of data file
    fig = plt.figure(figsize = (5, 6))
    gs = gridspec.GridSpec(4, 1)
    ax = fig.add_subplot(gs[:3])
    ax1 = fig.add_subplot(gs[-1])
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    ax.set_ylabel('C(k)', fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax1.minorticks_on()
    ax1.tick_params('x', which='major', direction='in', length=3)
    ax1.tick_params('y', which='major', direction='in', length=3)
    ax1.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax1.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    ax1.set_xlabel('k', fontsize = 13)
    ax1.set_ylabel('C(k)', fontsize = 13)
    ax1.grid(alpha=0.3)
    ax1.set_axisbelow(True)

    list_eta = []
    for file in onlyfiles:
        identificative = file.split('_Gap_energy_')[0]
        json_file = [f for f in jsonfiles if identificative in f][0]
#        fig, ax = plt.subplots(1,1)
        data_dict = extract_from_json(json_dir + json_file)
        eta = data_dict["eta"]
        list_eta.append(eta)
    list_eta = np.array(list_eta)
    N = 20/list_eta
    m = N >= 15
    list_eta = list_eta[m]
    onlyfiles = np.array(onlyfiles)[m]
    order_eta = np.argsort(list_eta)
    onlyfiles = onlyfiles[order_eta][::2]
    list_eta = list_eta[order_eta][::2]
    print(len(list_eta))
    cmap = plt.cm.get_cmap('Spectral')
    color_num = np.linspace(0, 1, len(list_eta))
    for file, eta, n in zip(onlyfiles, list_eta, color_num):
        data = np.loadtxt(data_dir + file)
        x = data[:, 0]
        y = data[:, 1]
        dy = data[:, 2]
        fit_data(x, y, dy, ax = ax, eta = np.round(eta, 3), c = cmap(n), lw = 0.8)
        fit_data(x, y, dy, ax = ax1, eta = np.round(eta, 3), c = cmap(n), lw = 0.7)
    ax.legend()
    ax1.set_yscale('log')
    ax.set_title('Andamento di $C(k)$ al variare di $k$', fontsize = 13)
    plt.savefig(f'../figure/fit_esponenziali_Delta_E{gap}.png', dpi = 200)
    plt.show()
    
def plot_results(folder_parent = 'beta_omega_20', gap = 1, kind = ''):
    kind_1 = ''
    if kind == 'sconnessa': 
        kind = '_solo_sconnessa'
        kind_1 = '_sconnessa'
    def fit_func(x, m, q):
        return m * x + q
    data_dir = '../dati/lim_continuum_E1_E0' + kind + '/' + folder_parent + '/'
    data_file = 'limite_continuo_' + str(gap) + kind_1 + '.dat'
    x, y, dy = np.loadtxt(data_dir + data_file, unpack=True)
    init = [0, 0]
    w = 1/dy**2
    pars, covm = curve_fit(fit_func, x, y, init, dy)
    chi = ((w*(y-fit_func(x,*pars))**2)).sum()
    ndof = len(y) - len(init)
    errors = np.sqrt(np.diag(covm))
    xx = np.linspace(np.min(x), np.max(x), 1000)
    DE_cont, dDE_cont = pars[1], errors[1]
    print('Delta E1 continuom:', DE_cont, '+-', dDE_cont)
    print('chi_red = ', chi/ndof)
    fig, ax = plt.subplots(1,1)
    ax.errorbar(x, y, yerr = dy, fmt = '.', label = 'Dati')
    ax.plot(xx, fit_func(xx, *pars), label = 'Fit')
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    y_label = ''
    if gap == 1: y_label = fr'$\Delta E_1$'
    if gap == 2: y_label = fr'$\Delta E_2$'
    ax.set_ylabel(y_label, fontsize = 13)
    ax.set_xlabel(fr'$\eta^2$', fontsize = 13)
    title_gap = ''
    if gap == 1: title_gap = r'$\Delta E_1$'
    if gap == 2: title_gap = r'$\Delta E_2$'
    ax.set_title('Limite al continuo per ' + title_gap, fontsize = 13)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    plt.legend()
    plt.savefig(f'../figure/limite_continuo_{gap}.png', dpi = 200)
    plt.show()
    return

if __name__ == '__main__':
    folder_parent = 'beta_omega_20'
    sconn = True
    gap = 1
    if sconn == True:
        kind = 'sconnessa'
    else:
        kind = ''
#    elaborate_all(folder_parent, kind = kind)
    plot_results(folder_parent, gap = gap, kind = kind)
    plot_exponential(folder_parent, gap = gap, kind = kind)

