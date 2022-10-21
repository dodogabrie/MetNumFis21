import time
import numpy as np
import math
cimport numpy as np # Make numpy work with cython
cimport cython
from libc.math cimport exp

def do_calc(int N, int iflag, int measures, int i_term,
            int i_decorrel, double extfield,
            double beta, int save_data = 1, int save_lattice = 1):
    """
    Main function for the ising model in 2D:
    Evaluate the energy and the magnetization of a "lattice of spin".
    Parameters
    ----------
    N : integer
        Linear dimension of the matrix
    iflag : integer
        If zero start the simulation "cold", all the spin up.
        If one start the simulation at T infinite, random spin.
    measures : integer
        Number of evaluation of energy and magnetization.
    i_decorrel : integer
        Number of iteration of the metropolis to decorrelate.
    extfield : float
        External magnetic field.
    beta : float
        Physics quantity defined by 1/(Kb*T) with T the standard temperature.
    save_data : bool
        Save magnetization and energy in a txt file
    save_lattice : bool
        Save the lattice in a txt file
    Results
    -------
    (numpy 1d array, numpy 1d array)
    Sampled Magnetization and Energy in two array of lenght "measures".
    """
    # define the random generator
    rng =  np.random.Generator(np.random.PCG64())
    # geometry array
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] npp = np.zeros(nlat).astype(int)
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] nmm = np.zeros(nlat).astype(int)
    # random array
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] rr = np.empty(3*i_decorrel*nlat*nlat)
    # lattice array
    cdef np.ndarray[np.int_t, ndim=2, mode='c'] y = np.empty((N, N)).astype(np.double)

    cdef int i, idec # index for loops

    geometry(nlat, npp, nmm) # Set the boundary conditions
    inizialize_lattice(iflag, nlat, field) # Inizialize the lattice

    for i in range(i_term):
        update_metropolis(y, N, npp, nmm, beta, extfield, rr, idec*3*nlat*nlat)

    for i in range(measures):
        rr = rng.uniform(size = 3*i_decorrel*nlat*nlat) # Extract the random points for the MC
        for idec in range(i_decorrel):
            update_metropolis(field, nlat, npp, nmm, beta, extfield, rr, idec*3*nlat*nlat) # MC
        magn[i] = magnetization(field, nlat, nvol) # Compute magnetization
        ene[i]  = energy(field, extfield, nlat, nvol, npp, nmm) # Compute energy
    if save_data:
        np.savetxt(f'../data/nlat{nlat}/data_beta{beta}_nlat{nlat}.dat', np.column_stack((magn, ene))) # Save Energy and Magnetization
    if save_lattice:
        np.savetxt(f'../data/lattice_matrix/lattice_nlat{nlat}_beta{beta}', field,
                   fmt='%0.f', header = f'L: {nlat}, beta: {beta}, measures : {measures}, i_decorrel : {i_decorrel}, iflag : {iflag}') # Save the lattice in file
    return magn, ene

@cython.boundscheck(False)  # Deactivate bounds checking ---> big power = big responsability
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void geometry(int nlat, np.int_t[:] npp, np.int_t[:] nmm):
    """
    Set the boundary conditions on the lattice: in this case the conditions
    are periodic.
    """
    cdef int i
    for i in range(nlat):
        npp[i] = i + 1
        nmm[i] = i - 1
    npp[nlat-1] = 0
    nmm[0] = nlat-1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void inizialize_lattice(int iflag, int nlat, np.double_t[:] y):
    """
    Inizialize lattice values (to 1 or -1 like the spin up/down)
    """
    rng =  np.random.Generator(np.random.PCG64())

    cdef double x
    cdef int i

    if iflag == 0: # Cold start --> All spin up
    for i in range(nlat):
        y[i] = 0.
    elif iflag == 1: # Warm start --> Random spin
        for i in range(nlat):
            x = 1. - 2*rng.uniform()
            y[i] = x
    else:
        y = np.loadtxt('path.dat').astype(np.double)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Activate C division
cdef double magnetization(np.int_t[:,:] field, int nlat, int nvol):
    """
    Compute the magnetization of the system as (sum of the spin)/Volume
    """
    cdef int i, j
    cdef double xmagn = 0.
    for i in range(nlat):
        for j in range(nlat):
            xmagn = xmagn + field[i, j]
    return xmagn/float(nvol)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cdef double energy(np.int_t[:,:] field, double extfield, int nlat, int nvol,
                   np.int_t[:] npp, np.int_t[:] nmm):
    """
    Compute the energy of the system as
        E = [0.5 * sum(neighboor * spin) - sum(extfield * spin)]/Volume
    """
    cdef int i, j, force
    cdef double xene = 0.
    for i in range(nlat):
        for j in range(nlat):
            force = neigh_force(i, j, field, npp, nmm)
            xene = xene - 0.5 * force * field[i, j]
            xene = xene - extfield * field[i, j]
    return xene/float(nvol)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline void update_metropolis(np.double_t[:] y, # the path
                                   int N, # size of path
                                   double eta,
                                   double d_metro,
                                   np.double_t[:] rr,
                                   np.double_t[:] logrr,
                                   int[:] npp,
                                   int[:] nmm,):
    cdef double force, y_real, y_test
    cdef double c1 = 1./eta
    cdef double c2 = c1 + eta/2.
    cdef int i, ip, im
    for i in range(N):
        ip = npp[i]
        im = nmm[i]
        force = y[ip] + y[im]
        y_real = y[i]
        y_test = y_real + 2 * d_metro * (0.5 - rr[i])

        p_rat = c1 * y_test * force - c2 * y_test * y_test
        p_rat += - c1 * y_real * force + c2 * y_real * y_real

        if logrr[i] < p_rat: y[i] = y_test


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int neigh_force(int i, int j, np.int_t[:,:] field, np.int_t[:] npp, np.int_t[:] nmm):
    """
    Compute the neighboors force
    """
    cdef int ip, im, jp, jm
    ip = npp[i]
    im = nmm[i]
    jp = npp[j]
    jm = nmm[j]
    return field[i, jp] + field[i, jm] + field[ip, j] + field[im, j]
