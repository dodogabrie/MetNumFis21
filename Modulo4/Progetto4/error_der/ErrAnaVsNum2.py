"""
This file evaluate the error in different numerical derivative methods
varying the grid step size dx.
"""
import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import NumDerivative as der

def err1(f_ana, f_num):
	f_diff = np.abs(f_ana - f_num)
	return np.sum(f_diff) / len(f_diff)

###############################################################
#define input variable
L = 10
N = 100
nu = 0.01

###############################################################
#define the dicrete interval dx
dx = L/N
x = np.arange(0,L,dx)

###############################################################
#evaluate the function u 
def u(x):
    return np.sin(2*np.pi*x/L)

##########################DERIVATIVE############################
#analitic derivative

def u2(x):
    return - (2*np.pi/L)**2 * np.sin(2*np.pi*x/L)
    
###########################ERROR#################################

Err1 = err1(u2(x), der.forward_der(u(x), dx))
#print(Err1)

##########################SCALING dx##############################

L = 10
nu = 0.01
list_N = []
list_dx = []
list_err1_s2 = []
list_err1_fft2 = []
list_err1_dfc2 = []
NN = np.logspace(1, 3, 50).astype(int)
for N in NN:
	dx = L / N
	x = np.linspace(0,L,N, endpoint = False)
	f_ana = u2(x)
	# symmetric error
	f_num = der.simm_der2(u(x), dx)
	list_err1_s2.append(err1(f_ana, f_num))
	# fft error
	f_num = der.fft_der2(u(x), dx)
	list_err1_fft2.append(err1(f_ana, f_num))
    # diff fin comp 
	f_num = der.diff_fin_comp_der2(u(x), dx)
	list_err1_dfc2.append(err1(f_ana, f_num))
	list_N.append(N)
	list_dx.append(dx)

fig, ax = plt.subplots(1, 1, figsize = (8, 8))

ax.plot(list_N, list_err1_s2, marker = '.', label = 'simm', lw = 0.6) #error of order dx**2
ax.plot(list_N, list_err1_fft2, marker = '.', label = 'fft', lw = 0.6) #error of order very low
ax.plot(list_N, list_err1_dfc2, marker = '.', label = 'diff fin comp', lw = 0.6) #error of order dx**4
#ax.plot(list_dx, list_err2, marker = '.', c = 'brown') #bad 'cause not linear...
ax.grid(alpha = 0.3)
ax.loglog()
ax.minorticks_on()
ax.tick_params('x', which='major', direction='in', length=5)
ax.tick_params('y', which='major', direction='in', length=5)
ax.tick_params('y', which='minor', direction='in', length=3, left=True)
ax.tick_params('x', which='minor', direction='in', length=3, bottom=True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
ax.set_xlabel('N', fontsize = 15)
ax.set_ylabel('Errore', fontsize = 15)
title = "Errore della derivata seconda al variare di N"
ax.set_title(title, fontsize = 15)
plt.legend(fontsize = 12)
#plt.savefig('../figures/derivatives/err_varying_N.png', dpi = 200)
plt.show()
