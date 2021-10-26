"""
This module contains tools for PDE integration: Routine of integration and
simple function for visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
# The point is a relative import (relative to this folder)
from .derivate import simm_der, simm_der2, shift_test
import plotly.graph_objects as go

def euler(f, FHS, dt, param):
    """
    Simple euler integration step for PDE.

    Parameters
    ----------
    f : numpy nd array
        Array of initial value of function (it will be modified by euler,
        better to pass a copy of the intial values).
    FHS : python funciton
        Function that return the evaluation of the force of the system.
        The parameters of the function need to be: FHS(f, *param), where
        f is the first argument of euler, and *param are all the other
        parameters (passed in euler in a tuple).
    dt : float
        Temporal step size for the integrator.
    param : tuple
        Tuple containing the parameters of the function FHS.

    Returns
    -------
    numpy nd array
        Function f evaluated in this temporal step.
    """
    f[:] = f + FHS(f, *param) * dt
    return f

def RKN(f, N, FHS, dt, param):
    """
    Runge Kutta integration step for PDE.

    Parameters
    ----------
    f : numpy nd array
        Array of initial value of function (it will be modified by euler,
        better to pass a copy of the intial values).
    N : int
        Runge Kutta order.
    FHS : python funciton
        Function that return the evaluation of the force of the system.
        The parameters of the function need to be: FHS(f, *param), where
        f is the first argument of euler, and *param are all the other
        parameters (passed in euler in a tuple).
    dt : float
        Temporal step size for the integrator.
    param : tuple
        Tuple containing the parameters of the function FHS.

    Returns
    -------
    numpy nd array
        Function f evaluated in this temporal step.
    """
    g = np.copy(f)
    for k in range(N,0,-1):
        g = f + 1/k * FHS(g, *param) * dt
    f[:] = g
    return f

def plot_evolution(Method, f, x, tt, params, ax = None, time_to_plot=5):
    """
    Plot the evolution of a PDE given the method of integration and
    the equation to solve.

    Parameters
    ----------
    Method: python function
        Integration method (like RKN or euler).
    f : numpy nd array
        Array of initial value of function (it will be modified by euler,
        better to pass a copy of the intial values).
    x : numpy 1d array
        Spatial grid of the function f.
    tt : numpy 1d array
        Temporal grid containing each temporal step for the evaluation of f
        over time.
    N : int
        Runge Kutta order.
    FHS : python funciton
        Function that return the evaluation of the force of the system.
        The parameters of the function need to be: FHS(f, *param), where
        f is the first argument of euler, and *param are all the other
        parameters (passed in euler in a tuple).
    dt : float
        Temporal step size for the integrator.
    param : tuple
        Tuple containing the parameters of the method (first argument). Note
        that the parameter of 'method' contain also the parameter of RHS, for
        this reason the suggestion is to manage this argument like:

        >>> RHS_param = (...)
        >>> Meth_param = (...)
        >>> param = (*Meth_param, RHS_param)
        >>> plot_evolution(..., params, ...)

        In this way is more easier to control the problem.
    ax: matplotlib axes
        Axes in wich the results will be plotted (if None the function create a
        simple axes).
    time_to_plot : integer
        Number of plot to visualize (equally spatial choosen from tt array).

    Returns
    -------

    Example
    -------
    >>> def FHS(f, x, nu, L, dx, der2):
    >>>     return - np.sin(10 * np.pi * x/L) * f + nu * simm_der2(f, dx, der2)

    >>> # Parameters of the simulation
    >>> (Nt, dt, L, N, a, phi0, nu) # Passed somewhere
    >>> tmax = dt * Nt # Max simulation time
    >>> dx = L/N # Spatial grid step size
    >>> # Define the grid
    >>> x = np.linspace(0, L, N, endpoint = False)
    >>> tt = np.arange(0, tmax, dt)
    >>> # Initial condition
    >>> f0 = a * np.sin(2 * np.pi * x/L + phi0)
    >>> der2 = np.empty(len(x))

    >>> # Simulation of the system
    >>> RKorder = 4 # Runge kutta order
    >>> f = np.copy(f0) # Copy initial condition
    >>> FHS_params = (dx, der2) # Parameters of FHS
    >>> meth_params = (RKorder, test_wave, dt) # Parameters of method
    >>> params = (*meth_params, FHS_params) # All togheter
    >>> plot_evolution(RKN, f, x, tt, params, ax = ax, time_to_plot=5)
    >>> plt.legend()
    >>> plt.show()

    """
    if ax == None:
        fig, ax = plt.subplots()
    equally_div = int(len(tt)/time_to_plot)
    plot_instant = tt[::equally_div]
    j = 1
    ax.plot(x, f, lw = 2, c = 'green', label = f't = {tt[0]}')
    for i, t in enumerate(tt):
        Method(f, *params)
        if t == plot_instant[j]:
            ax.plot(x, f, alpha = 0.4, linewidth = 1, c = 'orange')
            if j < time_to_plot-1: j+=1
    ax.plot(x, f, lw = 2, c='r', label = f't = {tt[-1]}')

def surface_xt(func, x, t, uinit, param, imported_title = None, plot_dim = None):
    """
    Return a 2D surface of the solution of PDE in time.

    Parameters
    ----------
    func : function
        Function that evaluate the next temporal step of the solution u.
        Here you can pass the implementation of the LAX methods.
        The parameters of the function needs to be: func(u, *param) where
        u is the solution at time i and *param are the other parameters of the
        system (like alpha and ninner for the implemented LAX).
    x : 1d numpy array
        Spatial grid of the system.
    t : 1d numpy array
        temporal grid of the system.
    uinit : 1d numpy array.
        Initial condition of the PDE (a function evaluated on x).
        Nt : int
        Number of temporal step for the evolution of the PDE.
    param : tuple
        This values will be passed to func, for example you can put here
        (alpha, ninner) in the case of the LAX method.
    imported_title : string
        The title of the figure, if None the function will set this automatic.
    plot_dim : int
        In multidimensional case pass here the dimention to analize. If None
        the problem are assumed 1D.

    Returns
    -------
    """
    Nt = len(t)
    dt = t[1]-t[0]
    dx = x[1]-x[0]
    n = len(x)
    u = np.copy(uinit)
    X, Y = np.meshgrid(x, t)
    evo = np.empty((Nt, n))
    evo[0] = u[0]
    for i in range(Nt-1):
        u = func(u, *param)
        evo[i+1] = u[plot_dim]

    m = min(np.min(evo[-1]),np.min(uinit[plot_dim]))
    M = max(np.max(evo[-1]), np.max(uinit[plot_dim]))


    ## Define update functions #######
    def my_surface(evo, x, t, X, Y, dt, dx):
        startt = t[0]
        endt = t[-1]
        startx = x[0]
        endx = x[-1]
#        contoursy = {"show" : True, "start": startt, "end": endt, "size": dt, "width" : 1, "usecolormap" : True}
#        contoursx = {"show" : True, "start": startx, "end": endx, "size": 2 * dx, "width" : 1, "usecolormap" : True}
        startContourf = {"show" : True, "start": startt, "end": startt + dt, "size": dt, "width" : 1}
        return go.Surface( z=evo, x=X, y=Y, opacity = 0.7, colorscale = 'Viridis', contours = {'y' : startContourf})
    fig = go.Figure(data = my_surface(evo, x, t, X, Y, dt, dx))
    fig.update_layout(scene = dict(
                    zaxis = {'range' : [m - 0.5*np.abs(m), M + 0.5*np.abs(M)]},
                    xaxis_title='x',
                    yaxis_title='t',
                    zaxis_title='u'),
                    scene_camera_eye=dict(x=0, y=-2, z=0.6),
    )
    fig.show()


def test_diffusion(f, nu, dx, der2):
    """
    Testing the diffusion of a system (useless for control the
    quality of an algorithm). This function can be used as FHS for
    RKN and euler function.

    Parameters
    ----------
    f : numpy nd array
        Array of initial value of function (it will be modified by euler,
        better to pass a copy of the intial values).
    nu : float
        Diffusion parameter.
    dx : float
        Spatial step size for the derivative evaluation.
    der2 : numpy nd array
        Aux array for derivative evaluation.

    Returns
    -------
    numpy nd array
        Function f evaluated in this temporal step.
    """
    return nu * simm_der2(f, dx, der2)

def test_wave(f, dx, der2):
    """
    Testing the carryng of a wave by the system (useless for control the
    quality of an algorithm). This function can be used as FHS for
    RKN and euler function.

    Parameters
    ----------
    f : numpy nd array
        Array of initial value of function (it will be modified by euler,
        better to pass a copy of the intial values).
    dx : float
        Spatial step size for the derivative evaluation.
    der2 : numpy nd array
        Aux array for derivative evaluation.

    Returns
    -------
    numpy nd array
        Function f evaluated in this temporal step.
    """
    return - simm_der(f, dx, der2)
