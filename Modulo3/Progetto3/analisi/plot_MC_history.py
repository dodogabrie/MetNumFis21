import sys
sys.path.append('../../../utils/')
import numpy as np
import json
import m1.readfile as rf

def extract_from_json(file):
    json_file = file[:-4] + b'.json'
    with open(json_file, 'r') as f:
        datastore = json.load(f)
    return datastore


def get_U(file):
    data = rf.fastload(file, int(1e6))
    y2 = data[:,0]
    dy2 = data[:,1]
    datastore = extract_from_json(file)
    eta = datastore['eta']
    nlat = datastore['nlat']
    U = 1/(2*eta) + 1/2 * np.mean(y2) - 1/(2*eta**2)*np.mean(dy2)
    return U, eta, nlat

def show_history(file):
    data = rf.fastload(file, int(1e7))
    y2 = data[:,0]
    dy2 = data[:,1]
    return y2, dy2


if __name__ == '__main__':
    data_dir = '../dati/'
    eta = 1e-2
    inv_Neta = np.array([0.05, 0.333333, 0.5, 0.666667, 0.8, 1., 1.11111, 1.33333, 2., 2.5, 3.33333, 5, 10])
    nlat_list = (1/(inv_Neta*eta)).astype(int)
    U_list = []
    for nlat in nlat_list:
        file = data_dir + f'obs_nlat{nlat}/data_eta{eta}.dat'
        U, eta, nlat = get_U(file.encode('UTF-8'))
        U_list.append(U)
    U_list = np.array(U_list)
    nlat_list = np.array(nlat_list)
    xx = 1/(nlat_list*eta)

    import matplotlib.pyplot as plt
    plt.scatter(xx, U_list)
    plt.show()
#
#    nlat = 10
#    eta = 0.3
#    file = data_dir + f'obs_nlat{nlat}/data_eta{eta}.dat'
#    y2, dy2 = show_history(file.encode('UTF-8'))
#    import matplotlib.pyplot as plt
#    plt.plot(y2)
#    plt.show()


