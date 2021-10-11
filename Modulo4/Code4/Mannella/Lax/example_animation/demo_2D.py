"""
==========================
Rotating 3D wireframe plot
==========================

A very simple 'animation' of a 3D plot.  See also rotate_axes3d_demo.
"""

from __future__ import print_function

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import time

def generate(X, Y, phi):
    '''
    Generates Z data for the points in the X, Y meshgrid and parameter phi.
    '''
    R = 1 - np.sqrt(X**2 + Y**2)
    return np.exp(Y) * np.cos(2 * np.pi * X + phi) * R


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')

# Make the X, Y meshgrid.
xs = np.linspace(-1, 1, 70)
ys = np.linspace(-1, 1, 70)
X, Y = np.meshgrid(xs, ys)

# Set the z axis limits so they aren't recalculated each frame.
ax.set_zlim(-1, 1)

# Begin plotting.
wframe = None
tstart = time.time()

try:
    while True:
        print('Press Control-C to stop animation', end='\r')
        for phi in np.linspace(0, 180. / np.pi, 500):
            # If a line collection is already remove it before drawing.
            if wframe:
                ax.collections.remove(wframe)
        
            # Plot the new wireframe and pause briefly before continuing.
            Z = generate(X, Y, phi)
            wframe = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
            plt.pause(.0001)

except KeyboardInterrupt:
    print('Interrupted Correctly                  ')
    pass

print('\nAverage FPS: %f' % (100 / (time.time() - tstart)))
