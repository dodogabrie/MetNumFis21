import numpy as np

def euler_step(u, F, dt, *params):
    """
    Euler temporal step.

    Parameters
    ----------
    u: numpy 1d array
        The function at step n.
    F: function object
        Function that takes u as first parameter and arbitrary other 
        parameters.
    dt: float
        Temporal step of the integration.
    *params : auxiliar parameters
        Parameters of function F.

    Returns
    -------
    numpy 1d array
        Array with u at istant n+1

    """
    return u + dt * F(u, *params)

def RK2(u, F, dt, *params):
    """
    Runge Kutta of order 2 (time indipendent equations only).

    Parameters
    ----------
    u: numpy 1d array
        The function at step n.
    F: function object
        Function that takes u as first parameter and arbitrary other 
        parameters.
    dt: float
        Temporal step of the integration.
    *params : auxiliar parameters
        Parameters of function F.

    Returns
    -------
    numpy 1d array
        Array with u at istant n+1
    """
    order = 2 # Runge Kutta order
    u_copy = np.copy(u)
    for k in range(order, 0, -1):
        u_copy = u + 1/k * dt * F(u_copy, *params)
    u[:] = u_copy # just for animation algorithm
    return u_copy

def RKN(u, F, dt, N, *params):
    """
    Runge Kutta of order N (time indipendent equations only).

    Parameters
    ----------
    u: numpy 1d array
        The function at step n.
    F: function object
        Function that takes u as first parameter and arbitrary other 
        parameters.
    dt: float
        Temporal step of the integration.
    N: int
        Order of the Runge Kutta Method.
    *params : auxiliar parameters
        Parameters of function F.

    Returns
    -------
    numpy 1d array
        Array with u at istant n+1
    """
    u_copy = np.copy(u)
    for k in range(N, 0, -1):
        u_copy = u + 1/k * dt * F(u_copy, *params)
    u[:] = u_copy # just for animation algorithm
    return u_copy
