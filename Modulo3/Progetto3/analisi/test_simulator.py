import sys
sys.path.append('../../../utils/')
import numpy as np
import m1.readfile as rf

def test_x2(file):
    data = rf.fastload(file, int(1e7))
    x2 = data[:,0]
    print('n_data:', len(x2), 'mean:', np.mean(x2))


if __name__ == '__main__':
    file = b'../dati/nlat10/data_eta0.1.dat'
    test_x2(file)

