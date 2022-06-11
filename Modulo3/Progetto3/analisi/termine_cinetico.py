import sys
import matplotlib.pyplot as plt
sys.path.append('../../../utils/')
import numpy as np
import json
import m1.readfile as rf
from m1.error import err_mean_corr

def extract_from_json(file):
    json_file = file[:-4] + b'.json'
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
    list_K = [] # kinetic term
    list_dK = [] # kinetic term error
    list_U = []
    list_dU = []
    for nlat, eta in zip(nlat_list, eta_list):
        filename = f'../dati/potential_term/data_eta{eta}_nlat{nlat}.dat'
        print('getting data from file:', filename)
        y2, dy2 = get_data(filename)
        # Extract Kin energy
        K = 1/(2*eta) - dy2/(2*eta**2)
        list_K.append(np.mean(K))
        _, dK, _ = err_mean_corr(K)
        list_dK.append(dK)
        # Extract Pot energy
        list_U.append(np.mean(y2/2))
        _, dU, _ = err_mean_corr(y2/2)
        list_dU.append(dU)

    fig, ax = plt.subplots(1, 1)

    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=3)
    ax.tick_params('y', which='major', direction='in', length=3)
    ax.tick_params('y', which='minor', direction='in', length=1.5, left=True)
    ax.tick_params('x', which='minor', direction='in', length=1.5, bottom=True)
    ax.set_xlabel(r'$\eta$')
#    ax.set_ylabel(r'$\left<y^2\right>$')
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


    ax.errorbar(eta_list, list_K, yerr = list_dK, fmt = '.', label = r'$\frac{1}{2\eta} - \frac{\left<\Delta y^2\right>}{2\eta^2}$')
    ax.errorbar(eta_list, list_U, yerr = list_dU, fmt = '.', label = r'$\frac{\left<y^2\right>}{2}$')
    plt.legend(fontsize=17)
    ax.set_title("Contributo cinetico e potenziale all'energia interna")
    plt.savefig('../figure/contributi_energia_KU.png', dpi = 200)
    plt.show()
    # TODO: FIT DATA WITH f(x) = a + bx**2 and evaluate a (0 energy)


if __name__ == '__main__':
    get_fit_data()


