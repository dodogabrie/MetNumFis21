import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import roots

@njit(cache=True)
def f(x):
    return x**3/5 + x/5 + 0.1

@njit(cache=True)
def df(x):
    return 3/5 * x**2 + 1/5 

def find_root():
    i0, i1 = -1, 0
    max_step = 20
    bisec_step = 1
    root_m = 0
    root = roots.newton(f, df, i0, max_step)
    bis = []
    newt = []
    mix = []
    print('iter ', 'bisec   ', 'newton  ', 'mix')
    for k in range(1, max_step):
        root_b = roots.bisection(f, i0, i1, k)
        root_n = roots.newton(f, df, i0, k)
        root_m = roots.mixture1d(f, df, i0, i1, k, min(bisec_step, k))
        sep = 4 - (int(np.log10(k))+1)
        print(k,' '*sep, f'{root_b:.4f}',f' {root_n:.4f}', f' {root_m:.4f}')
        bis.append(np.abs(root_b - root))
        newt.append(np.abs(root_n - root))
        mix.append(np.abs(root_m - root))
    
    plt.plot(np.array(bis) , label = 'bisection' )
    plt.plot(np.array(newt), label = 'newton' )
    plt.plot(np.array(mix) , label = f'mix ({bisec_step} bisection iter)' )
    plt.legend()
    plt.show()

if __name__ == '__main__':
    find_root()
