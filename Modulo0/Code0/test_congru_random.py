import myrandom
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import time


def test_rand():
    show = True
    len_loop = 10000
    seed = 2   
    
    print(f'Number Generated: {len_loop}')
    #-------------------------DEFINE QUANTITIES---------------------------#
    
    # m = 10**8 + 1 # Lehmer
    # m = 2147483647 # 2^31 - 1  # Park-Miller 1988

    # a = 23     # original Lehmer implementation 1949 workin on ENIAC 
    # a = 16807  # Park-Miller 1988
    # a = 48271  # Park-Miller 1993

    # c = 0       # Both Lehmer and Park-Miller implementations

    
    x = np.empty(len_loop)
    y = np.empty(len_loop)


    #### Lehmer implementation ####
    m = 10**8 + 1
    a = 23
    c = 0
    start = time.time()
    x_L,y_L = myrandom.cong_rand_gen(np.copy(x), np.copy(y), seed, len_loop, a, c, m)
    
    #### Park-Miller implementation ####
    m = 2147483647
    a = 48271
    c = 0
    start = time.time()
    x_PM,y_PM = myrandom.cong_rand_gen(np.copy(x), np.copy(y), seed, len_loop, a, c, m)
    elaps = time.time()-start
    print(f'Generation time: {elaps:.6f}')

    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6))
        ax1.scatter(x_L,y_L, marker = '.', color = 'black')
        ax1.set_title('Lehmer', fontsize=18)

        ax2.scatter(x_PM,y_PM, marker = '.', color = 'black')
        ax2.set_title('Park-Miller', fontsize=18)
        plt.show()


if __name__=='__main__':
    test_rand()
