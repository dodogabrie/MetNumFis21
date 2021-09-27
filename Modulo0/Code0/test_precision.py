import integrators
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import time


@njit(cache=True)
def f(x, t, p = ()):
    return x * t

def test_int():

    real_benchmark = True
    plot = True
    x0 = 1
    ti = 0
    tf = 5
    h = 0.25

    t = np.arange(ti, tf + h, h)
    print('Num of teration:', len(t))
    x = np.empty( len(t) )
    x[0] = x0

    to_test = [integrators.euler, 
               integrators.trapezoid, 
               integrators.AB, 
               integrators.midpoint, 
               integrators.RK45]

    if real_benchmark:
        for method in to_test:
            method(f , np.copy(x) , t)
    
    tt = []
    xx = []
    for method in to_test:
        if method == integrators.RK45:
            y = np.copy(x[::2])
            tRK = t[::2]
            start = time.time()
            xx.append(method( f , y , tRK ))
            tt.append(time.time()-start)
        else:
            y = np.copy(x)
            start = time.time()
            xx.append(method( f , y , t ))
            tt.append(time.time()-start)

    x_eu, x_trap, x_AB, x_mid, x_RK = xx
    t_eu, t_trap, t_AB, t_mid, t_RK = tt

    if real_benchmark:
        print()
        print(f'Euler: {t_eu:.3f}s')
        print(f'Trapezoidal: {t_trap:.3f}s')
        print(f'AB: {t_AB:.3f}s')
        print(f'midpoint: {t_mid:.3f}s')
        print(f'RK45: {t_RK:.3f}s')

    if plot:
        fig, ax = plt.subplots(1,1,figsize=(8, 6))
        ax.plot( t,  x_eu   , marker='v', label = f'Euler')
        ax.plot( t,  x_trap , marker='*', label = f'Trapezoidal')
        ax.plot( t,  x_AB   , marker='P', label = f'AB')
        ax.plot( t,  x_mid  , marker='x', label = f'midpoint')
        ax.plot( tRK,  x_RK  , marker='o' , label = f'RK45')
        ax.plot(t, np.exp(t**2/2), label = 'True')

        ax.set_yscale('log')
        plt.legend()
        ax1 = ax.twinx()
        ax1.set_yscale('log')
        ax1.set_ylim(ax.get_ylim())
        ax1.tick_params(labelright=False)
        plt.savefig('../Figures/IntegratorPrecision.png', dpi = 100)
        plt.show()

if __name__=='__main__':
    test_int()
