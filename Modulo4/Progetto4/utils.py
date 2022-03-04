"""
FFT function, energy density function and plot results function
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


# Fourier transform in time 
def my_real_fft(u, dx):
    """
    Real Fast Fourier Transform.
    """
    N = len(u)
    # x : 2 * pi = y : 1 ---> unitary rate
    # => dy = dx/(2*pi)
    dy =  dx / (2 * np.pi)
    # fft(j) = (u * exp(-2*pi*i*j*np.arange(n)/n)).sum()
    fft = fftpack.fft(u) # Discret fourier transform 
    k = fftpack.fftfreq(N, dy) 
    return fft, k

# Fourier transform in time 
def my_real_ifft(u):
    """
    Real inverse Fast Fourier Transform.
    """
    # fft(j) = (u * exp(-2*pi*i*j*np.arange(n)/n)).sum()
    ifft = fftpack.ifft(u) # Discret fourier transform 
    return ifft

def evaluate_energy_density_spectrum(u_t, dx):
    # final fft
    fft_u, k_u = my_real_fft(u_t, dx) # FFT of final u 
    mod_u_k = np.abs(fft_u) # |u_k|
    mod_fft2_u = (mod_u_k)**2 # |u_k|^2
    mask_pos_k_u = k_u > 0 # mask for positive k final
    mod_fft2_u = mod_fft2_u[mask_pos_k_u]
    k_u = k_u[mask_pos_k_u]
    return mod_fft2_u, k_u

def plot_results(u, x, dx, label, axs = [None], **kwargs):
    if (axs == None).all():
        fig, axs = plot_template()
    u_k2, k = evaluate_energy_density_spectrum(u, dx)
    axs[0].plot(x, u, label = label, **kwargs)
    axs[1].scatter(k, u_k2, s = 10, label = label)
    return axs

def plot_template(row = 1, col = 2, figsize = (12, 3)):
    fig, axs = plt.subplots(row, col, figsize = figsize)
    ax = axs[0]
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('u', fontsize=15)
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
    ax = axs[1]
    ax.grid(alpha = 0.3)
    ax.minorticks_on()
    ax.tick_params('x', which='major', direction='in', length=5)
    ax.tick_params('y', which='major', direction='in', length=5)
    ax.tick_params('y', which='minor', direction='in', length=3, left=True)
    ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
    ax.set_xlabel('k', fontsize=15)
    ax.set_ylabel(r'$\left|u_k\right|^2$', fontsize=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return fig, axs

def check_CFL(c, dx, dt):
    f = c*dt/dx
    if f > 1: raise Exception(f'CFL condition not respected: c*dx/dt = {f}')
    else: print(f'CFL number: c*dt/dx = {f}')
    return

def check_VonNeumann(nu, dt, dx):
    f = nu * dt / (dx)**2
    if f > 1/2: print(f'Von Neumann on diffusion not respected: nu*dt/dx^2 = {f}')
    else: print(f'Von Neumann on diffusion: nu*dt/dx^2 = {f:.3e}')
    return


########Initial conditions##########################################################
def analytic_sol_u_sin_simple_diffusion(x, m, nu, T):
    L = x[1]-x[0] + x[-1]
    k = m * 2*np.pi/L
    sol = np.exp( -nu * k**2 * T) * np.sin(k*x)
    return sol

def analytic_sol_u_sin_simple_advection(x, m, c, T):
    L = x[1]-x[0] + x[-1]
    k = m * 2*np.pi/L
    sol = np.sin(k*(x-c*T))
    return sol

def analytic_sol_gaussian_advection(x, m, c, T, p):
    L = x[1]-x[0] + x[-1]
    k = m * 2*np.pi/L
    mu = p[0]
    min_pt = gaussian(0, p)
    sol = gaussian(x-(mu+c*T)%L + mu, p)
    pos_ref = np.argmin(np.abs(min_pt - sol))
    if pos_ref <= len(x)/2: 
        sol_old = gaussian(x-(mu+c*T)%L + mu + L, p)
        sol[:pos_ref] = sol_old[:pos_ref]
    else:
        sol_old = gaussian(x-(mu+c*T)%L + mu - L, p)
        sol[pos_ref:] = sol_old[pos_ref:]
    return sol

def u_sin_simple(x, p):
    L = p[0]
    m = p[1]
    return np.sin(m * 2*np.pi*x/L)# + np.sin(m2*np.pi*x/L)

def u_mix_sin(x, p):
    L = p[0]
    n_modes = p[1]
    u_init = 0
    for i in range(1, n_modes):
        u_init += u_sin_simple(x, L, m = i)
    return u_init

def gaussian(x, p):
    mu = p[0]
    sigma = p[1]
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(1/2*(x-mu)/sigma)**2)


########Equations##########################################################
def advection_diffusion(u, c, nu, dx, der1, der2):
    return - c * der1(u, dx) + nu * der2(u, dx)


