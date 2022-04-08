import numpy as np
import time
import matplotlib.pyplot as plt
import core.module as mymodule

def test():
    N = 100000000
    print(f'Generate {N} random number...')
    start = time.time()
    mymodule.fname(N)
    print(f'Exec. time Cython: {time.time()-start}')
    start = time.time()
    np.random.rand(N)
    np.random.rand(N)
    print(f'Exec. time Numpy: {time.time()-start}')

if __name__ == '__main__':
    test()
