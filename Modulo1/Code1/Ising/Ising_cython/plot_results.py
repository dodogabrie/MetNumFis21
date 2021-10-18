import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('data/data_obs_nlat10.dat')
x = data[:,0]
y = data[:,5]
plt.errorbar(x, y, yerr = data[:,6])
plt.show()
