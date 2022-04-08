"""
This module implement some numerical scheme for flux conservative equation:
    - Lax-Wendroff
"""
import numpy as np 

def simm_der2(u, dx):
    """
    Second symmetric derivative using (Mannella) periodic boundary conditions.
    (Simmetric finite difference method)
    """
    der = np.empty(len(u))
    der[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2])/(dx**2)
    der[0], der[-1] = der[-2], der[1]
    return der

def Lax_W_Two_Step(u, x, t, dt, dx, F, RHS, *args, **kwargs):
    """method that solves u(n+1), for the scalar conservation equation with source term:
        du/dt + dF/dx = RHS,
        where F = 0.5u^2 for the burger equation
        with use of the Two-step Lax-Wendroff scheme
        
        Args:
            u(array): an array containg the previous solution of u, u(n).
            t(float): time at t(n+1) 
        Returns:
            u[1:-1](array): the solution of the interior nodes for the next timestep, u(n+1).
    """
    ujm = u[:-2].copy() #u(j-1)
    uj = u[1:-1].copy() #u(j)
    ujp = u[2:].copy() #u(j+1)
    # LAX STEP
    up_m = 0.5*(ujm + uj) - 0.5*(dt/dx)*(F(uj,*args)-F(ujm, *args)) + 0.5*dt*RHS(t-0.5*dt, x[1:-1] - 0.5*dx, **kwargs)
    up_p = 0.5*(uj + ujp) - 0.5*(dt/dx)*(F(ujp, *args)-F(uj, *args)) + 0.5*dt*RHS(t-0.5*dt, x[1:-1] + 0.5*dx, **kwargs)
    # WENDROFF STEP
    u[1:-1] = uj -(dt/dx)*(F(up_p, *args) - F(up_m, *args)) + dt*RHS(t-0.5*dt, x[1:-1], **kwargs)

    u[0], u[-1] = u[-2], u[1]
    return u
