import numpy as np
import time
import matplotlib.pyplot as plt
import core.tridiag as td

def test():
    N = int(1e1)
    for i in range(100):
        diag = np.random.rand(N).astype(float)
        dlo  = np.random.rand(N-1).astype(float)
        dup  = np.random.rand(N-1).astype(float)
        b    = np.random.rand(N).astype(float)
        x_cy = np.empty(N)
        A = np.diag(diag) + np.diag(dlo, -1) + np.diag(dup, 1)

        start = time.time()
        x_cy, inv_cy = td.solve_tridiag(np.copy(diag), np.copy(dlo), np.copy(dup), np.copy(b))
        x_np = np.linalg.solve(A, b)
        inv_np = np.allclose(np.dot(A, x_np), b)
        if inv_cy and inv_np:
            if not np.allclose(x_np, x_cy):
                print(f'\nThe result with numpy is equal to mine? {np.allclose(x_np, x_cy)}')
                print(f'But my solution is a solution?? {np.allclose(np.dot(A, x_cy), b)}')
                if not np.allclose(np.dot(A, x_cy), b):
                    print(f'Tot diff: {np.sum(np.abs(np.dot(A, x_cy)-b))}\n')

if __name__ == '__main__':
    test()
