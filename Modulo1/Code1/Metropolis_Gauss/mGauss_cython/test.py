import m_gauss
import numpy as np
import matplotlib.pyplot as plt
import time 

def err(X):
    N = float(len(X))
    return np.sqrt( 1/N * 1/(N-1) * np.sum((X-np.mean(X))**2))

nstat = int(1e6)
start = 0
aver = 5.
sigma = 1.
delta = .1

start = time.time()
arr, acc = m_gauss.do_calc(nstat, start, aver, sigma, delta)
print(time.time()-start)

print(f'Mean: {np.mean(arr):.6f} +- {err(arr):.6f}')

plt.hist(arr, bins = 100)
plt.show()
