import numpy as np 
import matplotlib.pyplot as plt

def unit_imag(x, y):
    mnz = np.logical_or(x !=0, y!=0) # mask not zero
    z = x + 1j*y
    # divide by norm only number different from zero
    z[mnz] = z[mnz]/(np.sqrt(x[mnz]**2 + y[mnz]**2))
    return z

def G(x, y, r):
    # advection eq:
#    gain = 1 + r * ( unit_imag(x, y) - unit_imag(x, -y)) 
    # diff + adv with rc = rnu:
    gain = 1 - 2*r + 2*r*unit_imag(x, -y)
    return gain

def G_mod(x, y, *args):
    # x = Re
    # y = Im
    g = G(x, y, *args)
    return np.abs(g)

def plot_contourf(L, N, *args):
    X, Y = np.meshgrid(np.linspace(-2, 2, 512), np.linspace(-2, 2, 512))
    Z = G_mod(X, Y, *args)
#    Z[np.sqrt(X**2 + Y**2) > 1] = np.inf
    levels = np.array([0, 1])
    levels = np.linspace(0, 1 + 1e-9, 100)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(X, Y, Z, levels=levels, alpha = 0.7)
    plt.colorbar(im)
    circ = plt.Circle((0, 0), 1, fill = False)
    ax.add_artist(circ)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    dx = L/N
    kmax = np.pi / dx
    kmin = 2 * np.pi / L
    n_k = int(N/2-1)
    k = np.linspace(kmin, kmax, n_k, endpoint=False)
    kdx = k*dx
    x = np.cos(kdx)
    y = np.sin(kdx)
    ax.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    L = 10
    N = 100
    dx = 0.1
    dt = 0.1
    c = dx/dt
    c = c/3
    r = c * dt/(2*dx)
    plot_contourf(L, N, r)
    #plot_contourf_k(L, N, r)
