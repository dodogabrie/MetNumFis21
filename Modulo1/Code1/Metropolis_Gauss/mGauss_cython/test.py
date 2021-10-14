### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

import m_gauss
from error import err_corr, err
import numpy as np
import matplotlib.pyplot as plt
import time 

# Parameters of the system / Monte Carlo simulation
nstat = int(1e7+2000) # Num of measures
start_val = 0.   # Starting x 
aver = 0.        # Gaussian average
sigma = 1.       # Gaussian variance
delta = .1       # Parameter for the acceptance
kmax = 200       # Number of step for correlation (error)

# Inizialize the simulation
print( 'MC simulation: ...', end = '\r') # Print time
start = time.time() # For benchmark
arr, acc = m_gauss.do_calc(nstat, start_val, aver, sigma, delta)
print(f'MC simulation: {(time.time()-start):.3f}s') # Print time

# Remove the first nstat/4 values for termalization
cut = int(nstat/4)
arr = arr[cut:]
# Computing the acceptance
acceptance = np.sum(acc[cut:])/(nstat-cut)

# Extracting the error considering the correlation
print( 'Evaluatete error: ...', end='\r') # Time consuming operation...
start = time.time()
tau, e_corr, Ck = err_corr(arr, kmax)
print(f'Evaluatete error: {(time.time()-start):.3f}s\n') # Time consuming operation...

# Print the results
print(f'Mean (no corr): {np.mean(arr):.6f} +- {err(arr):.6f}')
print(f'Mean (  corr ): {np.mean(arr):.6f} +- {e_corr:.6f}\n')
print(f'Acceptance = {acceptance:.3f}')

# Show some results
fig, axs = plt.subplots(1, 3, figsize = (15, 5))

axs[0].hist(arr, bins = 50)
axs[0].set_xlabel('x')
axs[0].set_ylabel('count')
axs[0].set_title('Histogram of output')

axs[1].plot(arr, c='r')
axs[1].set_xlabel('MC step')
axs[1].set_ylabel('MC output')
axs[1].set_title('MC history')

axs[2].plot(Ck, c='k')
axs[2].set_xlabel('k')
axs[2].set_ylabel('C(k)')
axs[2].set_title('Correlation Function')

plt.tight_layout()
plt.show()
