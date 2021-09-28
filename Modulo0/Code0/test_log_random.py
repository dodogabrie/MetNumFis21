import myrandom
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import time


def test_rand():
    show = True
    len_loop = 100000
    seed = 0.9 
    
    print(f'Number Generated: {len_loop}')
    
    x = np.empty(len_loop)
    y = np.empty(len_loop)


    myrandom.log_map_gen(np.copy(x), np.copy(y), seed, len_loop)

    start = time.time()
    x, y = myrandom.log_map_gen(np.copy(x), np.copy(y), seed, len_loop)
    elaps = time.time()-start
    print(f'Generation time: {elaps:.6f}')

    if show:
        plt.scatter(x, y, marker = '.', color = 'black')
        plt.show()


if __name__=='__main__':
    test_rand()
