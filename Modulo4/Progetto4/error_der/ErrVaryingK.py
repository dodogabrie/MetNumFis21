"""
This file evaluate the error in different numerical derivative methods
varying the wavenumber k of an input sinusoid.
"""
import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import NumDerivative as der

# Define error
def err1(f_ana, f_num):
    """
    The error is the sum of the absolute values of the difference
    between the analytical and numerical solution.
    """
    f_diff = np.abs(f_ana - f_num)
    return np.sum(f_diff) / len(f_diff)

###############################################################
#evaluate the function u 
def u(x, k):
    w = 2 * np.pi / L
    return np.sin( k * (w * x) )

##########################DERIVATIVE############################
#analitic derivative

def u1(x, k):
    w =  2 * np.pi / L
    return k * w * np.cos( k * (w * x) )
 
##########################SCALING dx##############################

L = 10
N = 50000
nu = 0.01
dx = L/N
x = np.linspace(0,L,N, endpoint = False)
list_k = []
list_err1_b = []
list_err1_f = []
list_err1_s = []
list_err1_s_ord4 = []
list_err1_s_ord6 = []
list_err1_fft = []
list_err1_dfc = []
k_M = np.pi/dx # Nyquist K: max k (wave number) solved
#              # for grid with resolution dx
k_to_test = np.unique(np.logspace(0, 3, 50).astype(int))
for k in k_to_test:
    if k > k_M: # If k > k_M interrupt the loop
        print('Max k reached!!!')
        break
    # Analytical derivative
    f_ana = u1(x, k)
    # forward error
    f_num = der.forward_der(u(x, k), dx) # Derivative
    list_err1_f.append(err1(f_ana, f_num)) # append error
    # backward error
    f_num = der.backward_der(u(x, k), dx) # Derivative
    list_err1_b.append(err1(f_ana, f_num)) # append error
    # symmetric error
    f_num = der.simm_der(u(x, k), dx) # Derivative
    list_err1_s.append(err1(f_ana, f_num)) # append error
    # symmetric error order 4
    f_num = der.simm_der_Ord4(u(x, k), dx) # Derivative
    list_err1_s_ord4.append(err1(f_ana, f_num)) # append error
    # symmetric error order 6
    f_num = der.simm_der_Ord6(u(x, k), dx) # Derivative
    list_err1_s_ord6.append(err1(f_ana, f_num)) # append error
    # fft error
    f_num = der.fft_der(u(x, k), dx) # Derivative
    list_err1_fft.append(err1(f_ana, f_num)) # append error
    # diff fin comp error
    f_num = der.diff_fin_comp_der(u(x, k), dx) # Derivative
    list_err1_dfc.append(err1(f_ana, f_num)) # append error
    # Append k array
    list_k.append(k)


fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(list_k, list_err1_b, marker = 's', label = 'BackW', lw = 0.6) #error of order dx
ax.plot(list_k, list_err1_f, marker = '*', label = 'ForW', lw = 0.6) #error of order dx
ax.plot(list_k, list_err1_s, marker = '.', label = 'simm2', lw = 0.6) #error of order dx**2
ax.plot(list_k, list_err1_s_ord4, marker = '.', label = 'simm4', lw = 0.6) #error of order dx**2
ax.plot(list_k, list_err1_s_ord6, marker = '.', label = 'simm6', lw = 0.6) #error of order dx**2
ax.plot(list_k, list_err1_fft, marker = '.', label = 'fft', lw = 0.6) #error of order very low
ax.plot(list_k, list_err1_dfc, marker = '.', label = 'diff fin comp', lw = 0.6) #error of order dx**4
ax.grid(alpha = 0.3)
ax.loglog()
ax.minorticks_on()
ax.tick_params('x', which='major', direction='in', length=5)
ax.tick_params('y', which='major', direction='in', length=5)
ax.tick_params('y', which='minor', direction='in', length=3, left=True)
ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
ax.set_xlabel('k', fontsize = 15)
ax.set_ylabel('Errore', fontsize = 15)
title = "Errore della derivata prima al variare di k"
ax.set_title(title, fontsize = 15)
plt.legend(fontsize = 12)
#plt.savefig('figures/derivatives/err_varying_k.png', dpi = 200)
plt.show()
