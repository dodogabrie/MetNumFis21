import matplotlib.pyplot as plt
import numpy as np

#%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

# Data to esplore divided by lateral size of grid
nlats = [10, 20, 30, 40, 50, 60, 70, 80]

# Quantity to estimate
estimate = 'chi'
#estimate = 'ene'
#estimate = 'magn'
#estimate = 'c'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dict_title = {'chi': 'Suscettività', 'ene':'Energia', 'magn':'Magnetizzazione',
              'c':'Calore specifico'}

dict_name = {'chi': 'Suscettività', 'ene':'Energia', 'magn':'Magnetizzazione',
              'c':'Calore_specifico'}

dict_y = {'chi': '$\chi$', 'ene':'$\epsilon$', 'magn':'|M|','c':'C'}


# Extract list of data files
list_data = [f'../data/data_obs_nlat{n}_test_new.dat' for n in np.array(nlats)]
# Dictionary for indices of searched quantities
dict_val = dict(magn = [1, 2], ene = [3, 4], chi = [5, 6], c = [7, 8])

i, di = dict_val[estimate] # Extract indices from dict

# For every dataset (one for each nlat) plot the curve varing beta
fig, ax = plt.subplots(figsize=(6, 6))
color = iter(plt.cm.tab10(np.linspace(0, 1, len(nlats))))

for d, l in zip(list_data, nlats):
    data = np.loadtxt(d) # Extract data
    x = data[:,0] # X coordinate is always beta
    y = data[:,i]
    ax.errorbar(x, y, yerr = data[:,di], label = f'L = {l}', fmt='.',
                markersize = 3, c = next(color))

#---- Belluire ----------------------------------------------------------------
# Minor ticks and inner ticks
ax.minorticks_on()
ax.tick_params('x', which='major', direction='in', length=4)
ax.tick_params('y', which='major', direction='in', length=4)
ax.tick_params('y', which='minor', direction='in', length=2, left=True)
ax.tick_params('x', which='minor', direction='in', length=2,bottom=True)

# Ax font size
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

ax.grid(alpha=0.3)
ax.set_xlabel(r'$\beta$', fontsize=12)
ax.set_ylabel(rf'{dict_y[estimate]}', fontsize=12)
ax.set_title(f'{dict_title[estimate]} al variare di L', fontsize = 13)
plt.legend(fontsize=11)
#plt.savefig(f'../figures/estimator_varying_L/{dict_name[estimate]}', dpi=200)
plt.show()
