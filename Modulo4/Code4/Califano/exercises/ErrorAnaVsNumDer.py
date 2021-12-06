"""
This file evaluate the error in different numerical derivative methods
varying the grid step size dx.
"""

import numpy as np
import matplotlib.pyplot as plt
import NumDerivative as der

def err1(f_ana, f_num):
	f_diff = np.abs(f_ana - f_num)
	return np.sum(f_diff) / len(f_diff)


def err2(f_ana, f_num):
	f_diff = f_ana - f_num
	return np.sqrt(np.sum(f_diff**2)) / len(f_diff)



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

def u1(x):
    return 2*np.pi/L * np.cos(2*np.pi*x/L)
    
###########################ERROR#################################

Err1 = err1(u1(x), der.forward_der(u(x), dx))
Err2 = err2(u1(x), der.forward_der(u(x), dx))
#print(Err1, Err2)

##########################SCALING dx##############################

L = 10
nu = 0.01
list_N = []
list_dx = []
list_err1_b = []
list_err1_f = []
list_err1_s = []
list_err1_fft = []
list_err1_dfc = []
list_err1_dfc2 = []
list_err2 = []
for N in range(10,1000,5):
	dx = L / N
	x = np.linspace(0,L,N, endpoint = False)
	f_ana = u1(x)
	# forward error
	f_num = der.forward_der(u(x), dx)
	list_err1_f.append(err1(f_ana, f_num))
	list_err2.append(err2(f_ana, f_num))
	# backward error
	f_num = der.backward_der(u(x), dx)
	list_err1_b.append(err1(f_ana, f_num))
	# symmetric error
	f_num = der.simm_der(u(x), dx)
	list_err1_s.append(err1(f_ana, f_num))
	# fft error
	f_num = der.fft_der(u(x), dx)
	list_err1_fft.append(err1(f_ana, f_num))
    # diff fin comp
	f_num = der.diff_fin_comp_der(u(x), dx)
	list_err1_dfc.append(err1(f_ana, f_num))

    # diff fin comp 2
	f_num = der.diff_fin_comp_der2(u(x), dx)
	list_err1_dfc2.append(err1(f_ana, f_num))
	list_N.append(N)
	list_dx.append(dx)
	
plt.plot(list_dx, list_err1_f, marker = '.', label = 'ForW') #error of order dx
plt.plot(list_dx, list_err1_b, marker = '.', label = 'BackW') #error of order dx
plt.plot(list_dx, list_err1_s, marker = '.', label = 'simm') #error of order dx**2
plt.plot(list_dx, list_err1_fft, marker = '.', label = 'fft') #error of order very low
#plt.plot(list_dx, list_err1_dfc, marker = '.', label = 'diff fin comp.') #error of order dx**4
plt.plot(list_dx, list_err1_dfc2, marker = '.', label = 'diff fin comp. 2') #error of order dx**4
#plt.plot(list_dx, list_err2, marker = '.', c = 'brown') #bad 'cause not linear...
plt.grid(alpha = 0.3)
plt.xlabel('dx')
plt.ylabel('Err1')
plt.legend()
plt.show()
