import m_gauss
import numpy as np
import matplotlib.pyplot as plt
import time 

def dfi_dfik(x, k, N, mean_x):
    return (x[ 0 : N-k ] - mean_x) * (x[ k :  N  ] - mean_x)

def C(x, kmax):
    N = len(x)
    Ck = np.empty(kmax-1)
    mean_x = np.mean(x)
    Ck = [1/(N-k) * np.sum(dfi_dfik(x, k, N, mean_x)) for k in range(1, kmax)]
    return Ck

def err_corr(x, kmax):
    Ck = C(x, kmax)
    tau = np.sum(Ck)
    print(tau)
    return err(x) * np.sqrt(1 + 2*tau), Ck
    

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

arr = arr[int(nstat/4): ]

kmax = 2000
start = time.time()
e_corr, Ck = err_corr(arr, kmax)
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
