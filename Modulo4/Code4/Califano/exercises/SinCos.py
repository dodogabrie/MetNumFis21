import numpy as np
import matplotlib.pyplot as plt


#####define input variable
L = 10
N = 100
nu = 0.01

#####define the dicrete interval dx
dx = L/N
x = np.arange(0,L,dx)

#####evaluate the function u and the analitic derivative
def u(x):
    return np.sin(2*np.pi*x/L)

def u1(x):
    return 2*np.pi/L * np.cos(2*np.pi*x/L)

#####save results in file txt
np.savetxt('SinCos.txt', np.column_stack((x,u(x),u1(x))),
            header = 'x    u   u1', fmt = '%.4f')

#####verify periodicity
#plt.plot(np.concatenate((x,x+L)),np.concatenate((u(x), u(x+L))))
#plt.show()

#####plot function u and u1
#plt.plot(x,u(x))
#plt.plot(x,u1(x))
#plt.show()

#####numerical derivative
def foward_der(u,dx):
    """
    Derivative considering the next point
    """
    der = np.empty(len(u))
    der[:-1] = (u[1:] - u[:-1])/dx
    der[-1] = der[0]
    return der

def backward_der(u,dx):
    """
    Derivative considering the previous point
    """
    der = np.empty(len(u))
    der[1:] = (u[1:] - u[:-1])/dx
    der[0] = der[-1]
    return der

plt.plot(x, u1(x), color = 'blue')
plt.plot(x, foward_der(u(x), dx), color = 'red')
plt.plot(x, backward_der(u(x), dx), color = 'green')
plt.show()
