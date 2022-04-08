import numpy as np 
import matplotlib.pyplot as plt

def tanh_test(x, x0, L):
    return np.tanh((x-x0)/L)

if __name__ == "__main__":
    x = np.linspace(0, 10, 1000)
    LL = [0.01, 0.05, 0.1, 1]
    x0 = 5
    for L in LL:
        y = tanh_test(x, x0, L)
        plt.plot(x, y, label=f"L={L}")
    plt.legend()
    plt.show()
