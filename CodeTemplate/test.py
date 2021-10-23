import numpy as np
import time
import matplotlib.pyplot as plt
import core.module as mymodule

def test():
    N = int(1e2)

    start = time.time()
    x, sinx = mymodule.fname(N)
    print(f'Exec. time Cython: {time.time()-start}')
    #plt.plot(x, sinx)
    #plt.show()

if __name__ == '__main__':
    test()
