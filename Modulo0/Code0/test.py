import integrators
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import time

@njit
def f(x, t, p = ()):
    return x * t

def test_int():
    x0 = 1
    ti = 0
    tf = 5
    h = 0.25

    t = np.arange(ti, tf + h, h)
    print('Num of teration:', len(t))
    x = np.empty( len(t) )
    x[0] = x0


    start = time.time()
    x_eu   = integrators.euler( f , np.copy(x) , t )
    eu_t = time.time()-start

    start = time.time()
    x_trap = integrators.trapezoid( f , np.copy(x) , t )
    trap_t = time.time()-start

    start = time.time()
    x_AB   = integrators.AB( f , np.copy(x) , t )
    AB_t = time.time()-start

    start = time.time()
    x_mid  = integrators.midpoint( f , np.copy(x) , t )
    mid_t = time.time()-start

    start = time.time()
    x_RK  = integrators.RK45( f , np.copy(x) , t )
    RK_t = time.time()-start

    fig, ax = plt.subplots(1,1,figsize=(8, 6))
    ax.plot( t,  x_eu   , marker='v', label = f'Euler: {eu_t:.3f}s')
    ax.plot( t,  x_trap , marker='*', label = f'Trapezoidal: {trap_t:.3f}s')
    ax.plot( t,  x_AB   , marker='P', label = f'AB: {AB_t:.3f}s')
    ax.plot( t,  x_mid  , marker='x', label = f'midpoint: {mid_t:.3f}s')
    ax.plot( t,  x_RK  , marker='o', label = f'RK45: {RK_t:.3f}s')
    ax.plot(t, np.exp(t**2/2), label = 'True')

    ax.set_yscale('log')
    plt.legend()
    ax1 = ax.twinx()
    ax1.set_yscale('log')
    ax1.set_ylim(ax.get_ylim())
    ax1.tick_params(labelright=False)
    plt.savefig('../Figures/IntegratorPrecision.png', dpi = 200)
    plt.show()

if __name__=='__main__':
    test_int()
