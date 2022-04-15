import numpy as np
import matplotlib.pyplot as plt

def plot_lattice(filename):
    data = np.loadtxt(filename)
    plt.plot(data)
    return

if __name__ == "__main__":
    filename = "../dati/lattice_nlat50eta0.001.dat"
    plot_lattice(filename)
    filename = "../dati/lattice_nlat100eta0.001.dat"
    plot_lattice(filename)
    filename = "../dati/lattice_nlat1000eta0.001.dat"
    plot_lattice(filename)
    plt.show()
