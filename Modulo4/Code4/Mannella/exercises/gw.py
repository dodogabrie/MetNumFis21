### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################

import numpy as np
import matplotlib.pyplot as plt
import m4.animated_plot as aniplt # Tool for animated plots
from m4.PDE_tools import surface_xt # Tool for suface plots

def init_u(x, t0): # this is the first component of f
    """
    Initial function for the variable u.
    """
    return np.exp( - (x - t0)**2 / 2) # arbitrary choosen...

def sum_der_init_u(x, t0): # this is the second component of f
    """
    Derivative of initial function for the variable u:
        u_t + u_x
    """
    return np.zeros(len(x)) # The sum is zero choosing that init_u

def main():
    # Define problem variables
    s = 0
    v = 0

    dx = 0.1
    dt = 0.05
    N_step = 100
    n = 300
    
    # define the grid
    x = np.linspace( - n * dx, n * dx, 2 * n)

    u = init_u(x, 0) # First component of f
    sum_der_u = sum_der_init_u(x, 0) # Second component of f

    f_init = np.row_stack((u, sum_der_u)) # Stack the two components in one "vector"
    f = np.copy(f_init) # Copy to overwrite

    ###############Just Plot Part###########################################
#    # Temporal evolution
#    f = LAX(f, dx, dt, N_step, v, s)
#    # Plot results
#    plt.plot(x, f_init[0], label = 'initial')
#    plt.plot(x, f[0], label = 'final')
#    plt.legend()
#    plt.show()

    ###############Animated Plot Part#######################################
    n_inner = 10 # Step to jump
    t = np.linspace(0, n_inner * N_step * dt, N_step)
    lax_param = (dx, dt, n_inner, v, s) # Define paramters of LAX func in list
    #                                   # (Withou f)
    # Animated + sliders
#    aniplt.animated_with_slider(LAX, f, x, t, lax_param, plot_dim = 0)
    # Surface plots
#    surface_xt(LAX, f, x, t, lax_param, plot_dim = 0)
    return 

def lax(f, dx, dt, n_step, v, s):
    alpha = dt/dx*0.5 # define dt/(2dx)
    f1 = f[0] # first component of f
    f2 = f[1] # second component of f
    for i in range(n_step):
        f1[1:-1] = 0.5*(f1[2:] + f1[:-2]) + dt*f2[1:-1] - alpha*(f1[2:] - f1[:-2])
        f2[1:-1] = 0.5*(f2[2:] + f2[:-2]) + dt*(v*f1[1:-1] + s) - alpha*(-f2[2:] + f2[:-2])
        f[:, 0], f[:, -1] = f[:, -2], f[:, 1] # boundary condition (wait)
    return f

#def lax_wendroff(f, dx, dt, n_step, v, s):
#    alpha = dt/dx*0.5 # define dt/(2dx)
##    dt2 = dt * 0.5 # Half instant
#    f1 = f[0] # first component of f
#    f2 = f[1] # second component of f
#    f1_LAX = np.copy(f1)
#    f2_LAX = np.copy(f1)
#    for i in range(n_step):
#        # LAX step                                                        (      h          )
#        f1_LAX[:-1] = 0.5*(f1[1:] + f1[:-1]) - alpha*(f1[1:] - f1[:-1])  + dt*f2[:-1]
#        f2_LAX[:-1] = 0.5*(f2[1:] + f2[:-1]) - alpha*(-f2[1:] + f2[:-1]) + dt*(v*f1[:-1] + s)
#        # Leap Frog
#        f1[] = f1[] - 2*alpha*(f1_LAX[]-f1_LAX[])
#        f2[] = f2[] - 2*alpha*(f2_LAX[]-f2_LAX[])
#        f[:, 0], f[:, -1] = f[:, -2], f[:, 1] # boundary condition (wait)
#    return f


if __name__ == "__main__":
    main()
