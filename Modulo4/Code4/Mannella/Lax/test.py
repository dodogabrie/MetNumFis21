import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
import LAX as lax

def test_func(u):
    return np.exp( - u**2/2 ) * np.cos(2 * np.pi * u)

def test():

    v = 10
    dt = 0.002
    dx = 0.02
    Nt = 100
    n = 200
    ninner = 1

    alpha = v * dt /dx
    print(alpha)

    x = np.linspace(- n/2 * dx, n/2 * dx, n)
    u = test_func(x)
    
    #### Animation stuff
    def animate(i):
        # LAX goes directly here
        line.set_ydata( lax.LAX(u, alpha, ninner) )  # update the data
        return line,
    
    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,
    #####
    
    # Plot results
    fig, ax = plt.subplots()
    line, = ax.plot(x, np.sin(x))
    ani = animation.FuncAnimation(fig, animate, np.arange(1, Nt), init_func=init,
                                  interval=25, blit=True)
    plt.show()

if __name__ == '__main__':
    test()
