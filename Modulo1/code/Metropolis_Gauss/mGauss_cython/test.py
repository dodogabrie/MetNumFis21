import m_gauss
import numpy as np
import matplotlib.pyplot as plt
import time

def tau_int(x):
    N = len(x)
    k = int(N/8)
    mean_camp = np.mean(x)
    Ck = np.zeros(k)
    for i in range(1, k):
        sum = 0
        for j in range(1, N-k):
            sum += (x[j] - mean_camp) * (x[j+k] - mean_camp)
            Ck[i] = 1/(N-k) * sum
    tau = np.sum(Ck[0:k])
    return tau, Ck



def err(X, tau):
    N = float(len(X))
    err_nocorr = np.sqrt( 1/(N-1) * np.sum((X-np.mean(X))**2))
    err_corr = err_nocorr * np.sqrt((1+2*tau)/N)
    return err_nocorr, err_corr

nstat = int(1e4)
start = 0
aver = 5.
sigma = 1.
delta = .1

start = time.time()
x, acc = m_gauss.do_calc(nstat, start, aver, sigma, delta)
print(time.time()-start)

tau, Ck = tau_int(x)
print(f'Mean (nocorr): {np.mean(x):.6f} +- {err(x, tau)[0]:.6f}')
print(f'Mean (corr): {np.mean(x):.6f} +- {err(x, tau)[1]:.6f}')


fig, axs = plt.subplots(1,2)
axs[0].hist(x, bins = 100)
axs[0].set_xlabel('x')
axs[0].set_ylabel('count')
axs[1].plot(x)
axs[1].set_xlabel('MC_step')
axs[1].set_ylabel('MC_story')
plt.show()
