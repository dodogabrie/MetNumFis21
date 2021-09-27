""" Module for finding roots of any expression"""

from numba import jit_module
import numpy as np
import matplotlib.pyplot as plt

def bisection(f, i0, i1, max_step):
    """
    Bisection Methods for root.

    Parameters
    ----------
    f: function
        Function wich is the roots.
    i0: real value
        Initial value of the interval containig the root.
    i1: real value
        Final value of the interval containing the root. Must be
        greater than i0.
    max_step: integer value
        Number of step for the root finding algorithm.

    Returns
    -------
    real value
        The value of the root.
    """
    f0 = f(i0)
    f1 = f(i1)
    for i in range(max_step):
        root = (i0+i1)/2
        f_half = f(root)
        if f_half > 0: 
            i1 = root
            f1 = f(root)
        if f_half < 0: 
            i0 = root
            f0 = f(root)
        if f_half == 0: break
    return root 

def newton(f, df, init, max_step):
    """
    Newton Methods for finding roots.
    Parameters
    ----------
    f: function
        Function wich is the roots.
    df: function
        Derivative of function f.
    init: real value
        Initial value for the iteration.
    max_step: integer value
        Number of step for the root finding algorithm.

    Returns
    -------
    real value
        The value of the root.
    """
    for i in range(max_step):
        der = df(init)
        if der == 0: break
        root = init - f(init)/der
        init = root
    return root

def mixture1d(f, df, i0, i1, max_step, num_bisec):
    """
    Mix of the bisection method (first num_bisec iterations) and
    Newton method (last max_step - num_bisec iterations).

    Parameters
    ----------
    f: function
        Function wich is the roots.
    df: function
        Derivative of function f.
    i0: real value
        Initial value of the interval containig the root.
    i1: real value
        Final value of the interval containing the root. Must be
        greater than i0.
    max_step: integer value
        Number of step for the root finding algorithm.
    num_bisec: integer value
        Number of initial bisection step.

    Returns
    -------
    real value
        The value of the root.
    """

    root = bisection(f, i0, i1, num_bisec)
    if max_step > num_bisec:
        root = newton(f, df, root, max_step - num_bisec)
    return root 


def multi_newton(f, df, init, max_step):
    """
    Multidim. Newton method.

    Parameters
    ----------
    f: function
        Function wich is the roots. It has to return an array.
    df: function
        Derivative of function f. It has to return a matrix.
    init: real value
        Initial value for the iteration.
    max_step: integer value
        Number of step for the root finding algorithm.

    Returns
    -------
    real values
        The values of the roots.
    """

    for i in range(max_step):
        inv_df = np.linalg.inv(df(init))
        fi = f(init)
        delta = - np.dot( inv_df, fi )
        if der == 0: break
        root = init + delta
        init = root
    return root

jit_module(fastmath = True, nopython=True, cache = True)
