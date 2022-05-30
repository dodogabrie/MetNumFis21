import sys
sys.path.append('../../../utils/')
import numpy as np
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
    nlat_list = [5, 10, 15, 20, 25, 30, 60, 80, 100]
    eta_list = [0.6, 0.3, 0.2, 0.15, 0.12, 0.10, 0.05, 0.0375, 0.03]
    list_y2_mean = []
    list_y2_err = []
    for nlat, eta in zip(nlat_list, eta_list):
        filename = f'../dati/obs_nlat{nlat}/data_eta{eta}.dat'
        y2, _ = get_data(filename)
        list_y2_mean.append(np.mean(y2))
        _, dy2, _ = err_mean_corr(y2)
        list_y2_err.append(dy2)

    import matplotlib.pyplot as plt
    plt.errorbar(eta_list, list_y2_mean, yerr = list_y2_err, fmt = '.')
    plt.show()
    # TODO: FIT DATA WITH f(x) = a + bx**2 and evaluate a (0 energy)


if __name__ == '__main__':
    get_fit_data()


