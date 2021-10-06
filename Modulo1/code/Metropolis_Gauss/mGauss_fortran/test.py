import metro_gauss
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def test():
    nstat  = int(1e7)#110000
    start  = 0.0      
    aver   = 5.0      
    sigma2 = 1.0      
    delta  = 1        

    input_dat = [nstat, start, aver, sigma2, delta]
    in_str = ['nstat', 'start', 'aver', 'sigma2', 'delta']

    file='inputGauss'
    with open(file, 'w') as filetowrite:
        for i in range(len(input_dat)):
            filetowrite.write(f'{input_dat[i]}  !{in_str[i]}\n')

    print('Simulation...')
    start = time.time()
    metro_gauss.metrogauss()
    print(time.time()-start)
    
    print('Importing data...')
    _, x, acc= np.loadtxt('data.dat', unpack = True)

    mean = np.mean(x)
    dev = np.std(x)
    print(f'Mean = {mean:.2f}\nStd = {dev:.3f}')

    print('Plotting Results...')
    metro_gauss.metrogauss()
    plt.hist(x, bins = 100)
    plt.show()


if __name__ == '__main__':
    test()
