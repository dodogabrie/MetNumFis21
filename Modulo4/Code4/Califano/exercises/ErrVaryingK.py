"""
This file evaluate the error in different numerical derivative methods
varying the wavenumber k of an input sinusoid.
"""

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
N = 5000
nu = 0.01
dx = L/N
x = np.linspace(0,L,N, endpoint = False)
list_k = []
list_err1_b = []
list_err1_f = []
list_err1_s = []
list_err1_fft = []
list_err1_dfc = []
list_err1_dfc2 = []
k_M = np.pi/dx # Nyquist K: max k (wave number) solved
#              # for grid with resolution dx
for k in range(1, 60):
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
    # fft error
    f_num = der.fft_der(u(x, k), dx) # Derivative
    list_err1_fft.append(err1(f_ana, f_num)) # append error
    # diff fin comp error
    f_num = der.diff_fin_comp_der(u(x, k), dx) # Derivative
    list_err1_dfc.append(err1(f_ana, f_num)) # append error
    # diff fin comp error2
    f_num = der.diff_fin_comp_der2(u(x, k), dx) # Derivative
    list_err1_dfc2.append(err1(f_ana, f_num)) # append error

    # Append k array
    list_k.append(k)

plt.plot(list_k, list_err1_f, marker = '.', label = 'ForW') #error of order dx
plt.plot(list_k, list_err1_b, marker = '.', label = 'BackW') #error of order dx
plt.plot(list_k, list_err1_s, marker = '.', label = 'simm') #error of order dx**2
plt.plot(list_k, list_err1_fft, marker = '.', label = 'fft') #error of order very low
#plt.plot(list_k, list_err1_dfc, marker = '.', label = 'diff fin comp') #error of order dx**4
plt.plot(list_k, list_err1_dfc2, marker = '.', label = 'diff fin comp 2') #error of order dx**4
plt.grid(alpha = 0.3)
plt.xlabel('k')
plt.ylabel('Err1')
plt.legend()
plt.show()
