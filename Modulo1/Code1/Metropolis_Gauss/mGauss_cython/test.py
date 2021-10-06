import m_gauss
from error import err_corr, err
import numpy as np
import matplotlib.pyplot as plt
import time 

nstat = int(1e6)
start = 0
aver = 5.
sigma = 1.
delta = .1

start = time.time()
arr, acc = m_gauss.do_calc(nstat, start, aver, sigma, delta)
print(time.time()-start)

arr = arr[int(nstat/4): ]

kmax = 5000
start = time.time()
tau, e_corr, Ck = err_corr(arr, kmax)
print(time.time()-start)

print(f'Mean: {np.mean(arr):.6f} +- {err(arr):.6f}')
print(f'Mean: {np.mean(arr):.6f} +- {e_corr:.6f}')

fig, axs = plt.subplots(1, 2)

axs[0].hist(arr, bins = 50)
axs[0].set_xlabel('x')
axs[0].set_ylabel('count')

axs[1].plot(arr)
axs[1].set_xlabel('MC step')
axs[1].set_ylabel('MC story')

plt.show()
plt.plot(Ck)
plt.show()
