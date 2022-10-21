import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Testing the Hexagonal lattice created with python:
    - Start from a square matrix.
    - Transform the coordinates of the points in the matrix.
    - Keep the square matrix but change the number of nearest neighboors for 
      each point according on the Hexagonal shape.
    """
    nlat = 10
    hot = 1
    # initialize lattice
    if hot:
        rng = np.random.default_rng()
        field = rng.random((nlat, nlat))
        field[field > 0.5] = 1
        field[field <= 0.5] = -1
    if not hot:
        field = np.ones((nlat,nlat))

    # Initialize vector of diretions (periodic condition)
    npp = np.arange( nlat ) + 1
    npp[ nlat - 1 ] = 0
    nmm = np.arange( nlat ) - 1
    nmm[ 0 ] = nlat - 1
    # Choosing a point for tests
    i = 2
    j = 4
    print(f'Sum of spin on position ({i}, {j}):' , neigh_force(i, j, field, npp, nmm))
    plot_lattice(field, i, j)
    test_neighboors(nlat, npp, nmm, i, j)

########## Add to Ising #################################################
def neighboors(i, j, npp, nmm):
    """
    Evaluate the neighboors of a point in a hex. lattice.
    """
    ip = npp[i]
    im = nmm[i]
    jp = npp[j]
    jm = nmm[j]
    # Initialize output array (of coordinates)
    coordinates = np.empty((3, 2)).astype(int)
    #       -----------------------
    # i = 0 |          x     x    |
    #       |                     |
    # i = 1 | x     x           x |
    #       |                     |
    # i = 2 |          x     x    |
    #       |                     |
    # i = 3 | x     x           x |
    #       -----------------------

    if j % 2 == 0: coordinates[0] = np.array([i, jp])
    else: coordinates[0] = np.array([i, jm])

    if i % 2 == 0: # See a left shift
        coordinates[1] = np.array([ip, jm])
        coordinates[2] = np.array([im, jm])
    else: # See a right shift
        coordinates[1] = np.array([im, jp])
        coordinates[2] = np.array([ip, jp])
    return coordinates

def neigh_force(i, j, field, npp, nmm):
    """
    Evaluate the sum of the spin in the lattice.
    """
    neigh = neighboors(i, j, npp, nmm)
    force = 0
    for k in range(len(neigh)):
        i = neigh[k, 0]
        j = neigh[k, 1]
        force = force + field[i, j]
    return force
#################################################


def transform_lattice(field):
    """
    Transform the square matrix into a set of points X, Y on a Hexagon.
    """
    # Initialize the meshgrid of coordinates
    nlat = len(field)
    lat_grid = np.arange(nlat)
    X, Y = np.meshgrid( lat_grid, lat_grid)
    # Transform the X variable (rotation)
    X = X.astype(float)
    Y = Y.astype(float)
    ratio = np.sqrt(3)/2 # cos(60°)
    dil = 2
    j = 0
    for i in range(nlat):
        X[ ::2, i] += j
        X[1::2, i] += j
        if (i+1) % 2 == 0:
            j += 1
    X[1::2, :] += 3/2
    Y = Y * ratio
    return X, Y

def test_neighboors(nlat, npp, nmm, i, j):
    """
    Testing the function neighboors on the Hexagonal lattice.
    Plot the point (i, j) and his neighboors highlighted in the lattice.
    """
    field = np.ones((nlat, nlat))
    X, Y = transform_lattice(field)
    plt.figure(figsize=(10, 10))
    field[i, j] = 0
    for x, y in neighboors(i,j, npp, nmm):
        field[x, y] = 10
    plt.scatter(X[field==1], Y[field==1], marker = 'o', alpha = 0.1)
    plt.scatter(X[field==0], Y[field==0], marker = 'x')
    plt.scatter(X[field==10], Y[field==10], marker='H')
    plt.show()

def plot_lattice(field, i, j):
    """
    Plot the lattice with the spin up and down.
    """
    X, Y = transform_lattice(field)
    plt.figure(figsize=(14, 8))
    plt.scatter(X[field==1], Y[field==1], marker='^')
    plt.scatter(X[field==-1], Y[field==-1], marker='v')
    plt.scatter(X[i, j], Y[i, j], marker='s')
    plt.show()   

if __name__ == '__main__':
    main()
