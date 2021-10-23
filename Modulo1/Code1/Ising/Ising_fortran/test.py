import numpy as np

ii, magn, ene = np.loadtxt('data.dat', unpack = True)
def err(X):
    return np.sqrt(1/float(len(X))*1/(float(len(X))-1) * np.sum((X-np.mean(X))**2))

print(len(ene))
print(np.mean(ene), err(ene))
print(np.mean(magn), err(magn))

