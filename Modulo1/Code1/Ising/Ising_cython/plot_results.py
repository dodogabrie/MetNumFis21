import matplotlib.pyplot as plt
import numpy as np

nlats = [10,20, 30, 40, 50, 60]
list_data = [f'data/data_obs_nlat{n}_test.dat' for n in np.array(nlats)]
dict_val = dict(magn = [1, 2], ene = [3, 4], chi = [5, 6], c = [7, 8])
estimate = 'chi'
i, di = dict_val[estimate]
for d, l in zip(list_data, nlats):
    
    data = np.loadtxt(d)
    x = data[:,0]
    y = data[:,i]
    plt.errorbar(x, y, yerr = data[:,di], label = f'idec = 100, L {l}')
    
plt.legend()
plt.show()
