import matplotlib.pyplot as plt
import numpy as np

data10_test = np.loadtxt('data/data_obs_nlat10_test.dat')

dict_val = dict(magn = [1, 2], chi = [5, 6])

estimate = 'chi'

i, di = dict_val[estimate]

x = data10_test[:,0]
y = data10_test[:,i]
plt.errorbar(x, y, yerr = data10_test[:,di], label = 'idec = 100, L 10')

plt.legend()
plt.show()
