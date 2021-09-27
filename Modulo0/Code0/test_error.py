import integrators
from numba import njit
import numpy as np
import matplotlib.pyplot as plt

@njit(cache=True)
def f(x, t, p = ()):
    return x * t

def test_int():
    x0 = 1
    ti = 0
    tf = 5
    h = 0.025

    t = np.arange(ti, tf, h)
    print('Num of teration:', len(t))
    x = np.empty( len(t) )
    x[0] = x0


    e_eu   = integrators.richardson_error( f , np.copy(x) , t , method = 'euler')
    e_trap = integrators.richardson_error( f , np.copy(x) , t , method = 'trapezoid')
    e_AB   = integrators.richardson_error( f , np.copy(x) , t , method = 'AB')
    e_mid  = integrators.richardson_error( f , np.copy(x) , t , method = 'midpoint')
    e_RK   = integrators.richardson_error( f , np.copy(x) , t , method = 'RK45')

    fig, ax = plt.subplots(1,1,figsize=(8, 6))
    ax.plot( t[::2],  e_eu   , marker='v', label = f'Euler')
    ax.plot( t[::2],  e_trap , marker='*', label = f'Trapezoidal')
    ax.plot( t[::2],  e_AB   , marker='P', label = f'AB')
    ax.plot( t[::2],  e_mid  , marker='x', label = f'midpoint')
    ax.plot( t[::2],  e_RK  , marker='o', label = f'RK45')

    ax.set_yscale('log')
    plt.legend()
    plt.savefig('../Figures/IntegratorError.png', dpi = 100)
    plt.show()

if __name__=='__main__':
    test_int()
