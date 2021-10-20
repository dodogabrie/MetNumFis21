import matplotlib.pyplot as plt
import numpy as np

#%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

# Data to esplore divided by lateral size of grid
nlats = [10, 20, 30, 40, 50]

# Quantity to estimate 
estimate = 'chi'
#estimate = 'ene'
#estimate = 'magn'
#estimate = 'c'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Extract list of data files
list_data = [f'../data/data_obs_nlat{n}_test_new.dat' for n in np.array(nlats)]
# Dictionary for indices of searched quantities
dict_val = dict(magn = [1, 2], ene = [3, 4], chi = [5, 6], c = [7, 8])

i, di = dict_val[estimate] # Extract indices from dict

# For every dataset (one for each nlat) plot the curve varing beta
for d, l in zip(list_data, nlats):
    data = np.loadtxt(d) # Extract data
    x = data[:,0] # X coordinate is always beta
    y = data[:,i]
    plt.errorbar(x, y, yerr = data[:,di], label = f'idec = 100, L {l}', fmt='.')
plt.legend()
plt.show()
