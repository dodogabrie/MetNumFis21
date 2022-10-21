import sys
sys.path.append('../../../utils/')
import numpy as np
import json
from numba import njit, typed
import m1.readfile as rf
from m1.error import err_mean_corr, bootstrap_corr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def extract_from_json(file):
    json_file = file[:-4] + '.json'
    with open(json_file, 'r') as f:
        datastore = json.load(f)
    return datastore

def get_U(file):
    data = rf.fastload(file)
    y2 = data[:,0]
    dy2 = data[:,1]
    datastore = extract_from_json(file)
    eta = datastore['eta']
    nlat = datastore['nlat']
    U = 1/(2*eta) + 1/2 * np.mean(y2) - 1/(2*eta**2)*np.mean(dy2)
    return U, eta, nlat, y2, dy2

@njit
def U_func(arr, param):
    y2 = arr[:, 0]
    dy2 = arr[:, 1]
    eta = param[0]
    return 1/(2*eta) + 1/2 * np.mean(y2) - 1/(2*eta**2)*np.mean(dy2)

def extract_all_U(eta, nlat_list):
    data_dir = '../dati/U_varying_T/'
    U_list = []
    dU_list = []
    for nlat in nlat_list:
        print(nlat)
        file = data_dir + f'data_eta{eta}_nlat{nlat}.dat'
        U, eta, nlat, y2, dy2 = get_U(file)
        arr = np.column_stack((y2, dy2))
        err_U = bootstrap_corr(arr, U_func, param = typed.List([eta]))
#        _, err_y2, _ = err_mean_corr(y2)
#        _, err_dy2, _ = err_mean_corr(dy2)
#        err_U = np.sqrt((1/2 * err_y2)**2 + (1/(2*eta**2) * err_dy2)**2)
        U_list.append(U)
        dU_list.append(err_U)
    U_list = np.array(U_list)
    dU_list = np.array(dU_list)
    nlat_list = np.array(nlat_list)
    xx = 1/(nlat_list*eta)
    return xx, U_list, dU_list

def get_observables(file):
    data = rf.fastload(file, int(1e6))
    y2 = data[:,0] # <y^2> (media su reticolo)
    dy2 = data[:,1] # <dy^2> (media su reticolo)
    return y2, dy2

def internal_energy(x):
    # x = 1/(nlat*eta)
    return 1/2 + 1/(np.exp(1/x) - 1)

def fit_func(x, a):
    return a + 1/(np.exp(1/x) - 1)

def teo_vs_data(x, U_list, dU_list):
    xx = np.linspace(np.min(x), np.max(x), 1000)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    plt.plot(xx, internal_energy(xx), label='Teorico')
    plt.errorbar(x, U_list, yerr = dU_list, fmt = '.', label = 'Dati')
    plt.show()
    return

def get_zero_energy(x, U_list, dU_list):
    xx = np.linspace(np.min(x), np.max(x), 1000)
    sigma = dU_list
    w = 1/sigma**2
    init = [1/2]
    pars, covm = curve_fit(fit_func, x, U_list, init, sigma, absolute_sigma=False)
    chi = ((w*(U_list-fit_func(x,*pars))**2)).sum()
    ndof = len(U_list) - len(init)
    errors = np.sqrt(np.diag(covm))
    zero_ene = pars[0]
    zero_ene_err = errors[0]
    print(f'Zero energy: {zero_ene} +- {zero_ene_err}')
    print(f'Chi2 red: {chi/ndof}')
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

    ax.set_title("Energia interna al variare della temperatura", fontsize = 15)
    ax.set_ylabel('U', fontsize = 13)
    ax.set_xlabel(r'$\frac{1}{N\eta}$', fontsize = 13)


    ax.plot(xx, internal_energy(xx), label='Curva di fit')
    ax.errorbar(x, U_list, yerr = dU_list, fmt = '.', label = 'Dati')
    plt.legend()
    plt.savefig('../figure/U_varying_T.png', dpi = 200)
    plt.show()
    return



if __name__ == '__main__':
    data_dir = '../dati/fix_eta_varying_N/'
    eta = 1e-2
    nlat_list = [20, 25, 30, 40, 50, 70, 80, 100, 150, 200, 300, 400, 800]
#    xx, U_list, dU_list = extract_all_U(eta, nlat_list)
#    results = np.column_stack((xx, U_list, dU_list))
#    np.savetxt('../dati/U_vs_invNeta.dat', results)

    xx, U_list, dU_list = np.loadtxt('../dati/U_vs_invNeta.dat', unpack=True)
    m = np.ones(len(xx)).astype(bool)
    m[-2] = 0
    xx = xx[m]
    U_list = U_list[m]
    dU_list = dU_list[m]
#    teo_vs_data(xx, U_list, dU_list)
    get_zero_energy(xx, U_list, dU_list)
