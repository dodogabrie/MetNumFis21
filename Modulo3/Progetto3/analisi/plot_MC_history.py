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


def show_MC_history(file):
    data = rf.fastload(file, int(1e7))
    y2 = data[:,0]
    dy2 = data[:,1]
    datastore = extract_from_json(file)
    eta = datastore['eta']
    nlat = datastore['nlat']
    U = 1/2 * np.mean(y2) - 1/(2*eta)*np.mean(dy2)
    return U, eta, nlat

if __name__ == '__main__':
    file = b'../dati/obs_nlat10/data_eta0.001.dat'
    U_1, eta, nlat1 = show_MC_history(file)

    file = b'../dati/obs_nlat50/data_eta0.001.dat'
    U_2, eta, nlat2 = show_MC_history(file)
    
    file = b'../dati/obs_nlat100/data_eta0.001.dat'
    U_3, _, nlat3 = show_MC_history(file)

    file = b'../dati/obs_nlat500/data_eta0.001.dat'
    U_4, eta, nlat4 = show_MC_history(file)
 
    file = b'../dati/obs_nlat1000/data_eta0.001.dat'
    U_5, _, nlat5 = show_MC_history(file)

    U_list = np.array([U_1, U_2, U_3, U_4, U_5])
    nlat_list = np.array([nlat1, nlat2, nlat3, nlat4, nlat5])
    xx = 1/(nlat_list*eta)

    import matplotlib.pyplot as plt
    plt.scatter(xx, U_list)
    plt.show()


