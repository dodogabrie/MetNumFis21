import numpy as np
import time
import matplotlib.pyplot as plt
import core.tridiag as td

def test():
    N = int(1e3)
    Ntest = 100
    print(f'Started test of {Ntest} system of size {N}')
    print(f'The matrix coefficient are setted randomly in [0, 1]')
    tester = True
    mean_t_np = 0
    mean_t_cy = 0
    for i in range(Ntest):
        # Defining system random values
        diag = np.random.rand(N).astype(float)
        dlo  = np.random.rand(N-1).astype(float)
        dup  = np.random.rand(N-1).astype(float)
        b    = np.random.rand(N).astype(float)

        # Matrix of the system (for numpy)
        A = np.diag(diag) + np.diag(dlo, -1) + np.diag(dup, 1)

        # Starting routines
        start = time.time()
        x_cy, inv_cy = td.solve(np.copy(diag), np.copy(dlo), np.copy(dup), np.copy(b))
        mean_t_cy += time.time()-start

        start = time.time()
        x_np = np.linalg.solve(A, b)
        mean_t_np += time.time()-start

        # Checking the results
        inv_np = np.allclose(np.dot(A, x_np), b)
        if inv_np != inv_cy:
            print('Error in check if invertible')
            tester = False
        else:
            tester = tester and np.allclose(np.dot(A, x_cy), b)

    np_t = mean_t_np/Ntest
    cy_t = mean_t_cy/Ntest
    print(f'\nNumpy mean time: {np_t}')
    print(f'Cython mean time: {cy_t}')
    print(f'Mean speed up: {np_t/cy_t}')
    print(f'Accurated results? {tester}')

if __name__ == '__main__':
    test()
