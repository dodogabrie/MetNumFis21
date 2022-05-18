import numpy as np
import matplotlib.pyplot as plt

def plot_lattice(filename, label):
    data = np.loadtxt(filename)
    plt.plot(data, label = label)
    return

if __name__ == "__main__":
    filename = "../dati/lattice_nlat100/lattice_eta0.001.dat"
    plot_lattice(filename, label = '100')
    filename = "../dati/lattice_nlat200/lattice_eta0.001.dat"
    plot_lattice(filename, label = '200')
    filename = "../dati/lattice_nlat1000/lattice_eta0.001.dat"
    plot_lattice(filename, label = '1000')
    plt.legend()
    plt.show()
