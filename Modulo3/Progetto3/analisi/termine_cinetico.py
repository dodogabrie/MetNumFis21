import sys
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
    data = rf.fastload(file, int(1e6))
    y2 = data[:,0]
    dy2 = data[:,1]
    return y2, dy2

def get_fit_data():
    nlat_list = [5, 10, 15, 20, 25, 30, 60, 80, 100]
    eta_list = [0.6, 0.3, 0.2, 0.15, 0.12, 0.10, 0.05, 0.0375, 0.03]
    list_K = [] # kinetic term
    list_dK = [] # kinetic term error
    list_U = []
    list_dU = []
    for nlat, eta in zip(nlat_list, eta_list):
        filename = f'../dati/obs_nlat{nlat}/data_eta{eta}.dat'
        y2, dy2 = get_data(filename.encode('UTF-8'))
        # Extract Kin energy
        K = 1/(2*eta) - dy2/(2*eta**2)
        list_K.append(np.mean(K))
        _, dK, _ = err_mean_corr(K)
        list_dK.append(dK)
        # Extract Pot energy
        list_U.append(np.mean(y2/2))
        _, dU, _ = err_mean_corr(y2/2)
        list_dU.append(dU)

    import matplotlib.pyplot as plt
    plt.errorbar(eta_list, list_K, yerr = list_dK, fmt = '.')
    plt.errorbar(eta_list, list_U, yerr = list_dU, fmt = '.')
    plt.show()
    # TODO: FIT DATA WITH f(x) = a + bx**2 and evaluate a (0 energy)


if __name__ == '__main__':
    get_fit_data()


