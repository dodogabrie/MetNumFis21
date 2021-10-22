"""
Testing the sin(x) function in Cython and with numpy vectorial operation.
Evaluate also the derivative of sin(x) in 3 different way comparing them 
with the real analitic solution.
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

import numpy as np
import time
import matplotlib.pyplot as plt
#import core.Mysin as MS
from m4.derivate import foward_der, backward_der, simm_der

def test():
    Lx = 1
    N = int(1e5)

#### Cython implementation ###############################
#    start = time.time()
#    x, sinx = MS.MySin(Lx, N)
#    print(f'Exec. time Cython: {time.time()-start}')
#    plt.plot(x, sinx)
#    plt.show()
##########################################################


#### Numpy implementation  ###############################
    start = time.time()
    x = np.linspace(0, Lx * 2 * np.pi, N, endpoint = False)
    sinx = np.sin(x)
    print(f'Exec. time Numpy : {time.time()-start}')
#    np.savetxt(f'data/data.dat', np.column_stack((x, sinx))) # Save Energy and Magnetization
##########################################################

    plt.plot(x, sinx, label='func')
    der = np.copy(sinx)
    simm_der(sinx, der)
    plt.plot(x, der, label = 'der')
    plt.legend()
    plt.show()

### Derivate evaluation ##################################
#    lowN = int(1e2)
#    upN  = int(1e5)
#    step = lowN
#    e1, e2, e3 = [],[],[]
#    NN = np.arange(lowN, upN, step)
#    e1 = np.empty(len(NN))
#    e2 = np.empty(len(NN))
#    e3 = np.empty(len(NN))
#    i = 0
#    for n in NN:
#        x = np.linspace(0, Lx * 2 * np.pi, n)
#        sinx = np.sin(x)
#        e1[i] = np.mean(np.abs(np.cos(x) - foward_der(sinx)))
#        e2[i] = np.mean(np.abs(np.cos(x) - backward_der(sinx)))
#        e3[i] = np.mean(np.abs(np.cos(x) - simm_der(sinx)))
#        i+=1
#    plt.plot(NN, e1)
#    plt.plot(NN, e2)
#    plt.plot(NN, e3)
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.show()

if __name__ == '__main__':
    test()
