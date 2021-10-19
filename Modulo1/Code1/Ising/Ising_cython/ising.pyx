import time
import numpy as np
import math
cimport numpy as np # Make numpy work with cython
cimport cython
from libc.math cimport exp

def do_calc(int nlat, int iflag, int measures, 
            int i_decorrel, double extfield, 
            double beta, int save_data = 1):
    """
    Main function for the ising model in 2D:
    Evaluate the energy and the magnetization of a "lattice of spin". 

    Parameters
    ----------
    nlat : integer
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

    Results
    -------
    (numpy 1d array, numpy 1d array)
    Sampled Magnetization and Energy in two array of lenght "measures".
    """
    # define the random generator
    rng =  np.random.Generator(np.random.PCG64())
    # volume of the lattice
    nvol = nlat*nlat
    # geometry array
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] npp = np.zeros(nlat).astype(int)
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] nmm = np.zeros(nlat).astype(int)
    # results array
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] magn = np.empty(measures)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] ene = np.empty(measures)
    # random array
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] rr = np.empty(3*i_decorrel*nlat*nlat)
    # lattice array
    cdef np.ndarray[np.int_t, ndim=2, mode='c'] field = np.ones((nlat, nlat)).astype(int)

    cdef int i, idec # index for loops
    
    geometry(nlat, npp, nmm) # Set the boundary conditions
    inizialize_lattice(iflag, nlat, field) # Inizialize the lattice

    for i in range(measures):
        rr = rng.uniform(size = 3*i_decorrel*nlat*nlat) # Extract the random points for the MC
        for idec in range(i_decorrel):
            update_metropolis(field, nlat, npp, nmm, beta, extfield, rr, idec*i_decorrel) # MC
        magn[i] = magnetization(field, nlat, nvol) # Compute magnetization
        ene[i]  = energy(field, extfield, nlat, nvol, npp, nmm) # Compute energy
    if save_data:
        np.savetxt('lattice', field, fmt='%0.f') # Save the lattice in file
        np.savetxt(f'data/nlat{nlat}/data_beta{beta}_nlat{nlat}.dat', np.column_stack((magn, ene))) # Save Energy and Magnetization
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
cdef void inizialize_lattice(int iflag, int nlat, np.int_t[:,:] field):
    """
    Inizialize lattice values (to 1 or -1 like the spin up/down)
    """
    rng =  np.random.Generator(np.random.PCG64())

    cdef double x
    cdef int i, j

    if iflag == 0: # Cold start --> All spin up
        for i in range(nlat):
            for j in range(nlat):
                field[i, j] = 1

    if iflag == 1: # Warm start --> Random spin
        for i in range(nlat):
            for j in range(nlat):
                x = rng.uniform()
                if x > 0.5: field[i, j] = 1
                else: field[i, j] = -1

    if iflag != 0 or iflag != 1: # Previous history start
        field = np.loadtxt('lattice').astype(int)

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
cdef inline void update_metropolis(np.int_t[:,:] field, # the field
                                   int nlat, # lateral size
                                   np.int_t[:] npp, np.int_t[:] nmm, # geometry arrays
                                   double beta, double extfield, # simulation parameters
                                   np.double_t[:] rr, int skip): # random numbers parameters
    """
    Update the lattice with a metropolis.
    """
    cdef int ivol, i, j, phi
    cdef double force, p_rat
    for ivol in range(nlat*nlat):
        i = int(rr[skip + 3*ivol] * nlat)
        j = int(rr[skip + 3*ivol + 1] * nlat)

        force = beta * ( neigh_force(i, j, field, npp, nmm) + extfield )
        phi = field[i, j]
        
        p_rat = exp(-2. * phi * force)

        if rr[skip + 3*ivol + 2] < p_rat: field[i, j] = - phi

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
