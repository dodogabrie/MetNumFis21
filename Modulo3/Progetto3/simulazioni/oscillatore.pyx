import time
import numpy as np
import math
cimport numpy as np # Make numpy work with cython
cimport cython
from libc.math cimport exp, log

def simulator(double lattice, int seed, int nlat, int iflag, 
              int measures, int i_decorrel, int i_term, double d_metro,
              double eta, int save_data = 1, int save_lattice = 1):
    """
    Main function for the harmonic oscillator.
    Parameters
    ----------
    lattice : double
        The lattice of the system?
    seed: int
        The seed for the random number generator.
    nlat : integer
        Linear dimension of the lattice
    iflag : integer
        If zero start the simulation "cold", all the spin up.
        If one start the simulation at T infinite, random spin.
    measures : integer
        Number of evaluation of energy and magnetization.
    i_decorrel : integer
        Number of iteration of the metropolis to decorrelate.
    i_term: integer
        Number of iteration of the metropolis to reach the desired
        temperature?
    eta : double
        The lattice spacing (=wa)
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
    # results array
#    cdef np.ndarray[np.double_t, ndim=1, mode='c'] magn = np.empty(measures)
#    cdef np.ndarray[np.double_t, ndim=1, mode='c'] ene = np.empty(measures)
    # random array
    n_rand_metro = 2
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] rr = np.empty(n_rand_metro*i_decorrel*nlat)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] r_term = np.empty(n_rand_metro*i_term*nlat)
    # lattice array
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] field = np.ones(nlat)

    cdef int i, idec # index for loops

    geometry(nlat, npp, nmm) # Set the boundary conditions
    inizialize_lattice(iflag, nlat, field) # Inizialize the lattice

    r_term = rng.uniform(size = n_rand_metro*i_term*nlat) # Extract the random points for the MC
    for i in range(i_term):
        update_metropolis(field, nlat, npp, nmm, eta, r_term, i*n_rand_metro*nlat) # MC

    for i in range(measures):
        rr = rng.uniform(size = n_rand_metro*i_decorrel*nlat*nlat) # Extract the random points for the MC
        for idec in range(i_decorrel):
            update_metropolis(field, nlat, npp, nmm, beta, extfield, rr, idec*n_rand_metro*nlat) # MC
        magn[i] = magnetization(field, nlat, nvol) # Compute magnetization
        ene[i]  = energy(field, extfield, nlat, nvol, npp, nmm) # Compute energy
    if save_data:
        np.savetxt(f'../data/nlat{nlat}/data_beta{beta}_nlat{nlat}.dat', np.column_stack((magn, ene))) # Save Energy and Magnetization
    if save_lattice:
        np.savetxt(f'../data/lattice_matrix/lattice_nlat{nlat}_beta{beta}', field,
                   fmt='%0.f', header = f'L: {nlat}, beta: {beta}, measures : {measures}, i_decorrel : {i_decorrel}, iflag : {iflag}') # Save the lattice in file
    return magn, ene

#=================== FUNCTION TO DEFINE THE GEOMETRY =========================
@cython.boundscheck(False)  # Deactivate bounds checking 
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
#=============================================================================

#=================== FUNCTION TO INITIALIZE THE LATTICE ======================
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void inizialize_lattice(int iflag, int nlat, np.double_t[:] field):
    """
    Inizialize lattice values according to the iflag.
    """
    rng =  np.random.Generator(np.random.PCG64())

    cdef double x
    cdef int i

    if iflag == 0: # Cold start --> all site 0
        for i in range(nlat):
            field[i] = 0.
    else:
        if iflag == 1: # Warm start --> Random y
            for i in range(nlat):
                y = 1 - 2*rng.uniform() # Random number in (-1, 1)
                field[i] = y
        else:
            field = np.loadtxt('lattice')
#=============================================================================

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


#=================== FUNCTION FOR THE METROPOLIS STEP ========================
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline void update_metropolis(np.double_t[:] field, # the field
                                   int nlat, # lateral size
                                   np.int_t[:] npp, np.int_t[:] nmm, # geometry arrays
                                   double eta, # simulation parameters
                                   np.double_t[:] rr, int skip): # random numbers parameters
    """
    Update the lattice with a metropolis.
    """
    cdef double c1 = 1/eta
    cdef double c2 = c1 + eta/2
    cdef int i, ip, im
    cdef double force, p_rat, phi, phi_prova
    cdef double rand_num
    for i in range(nlat):
        ip = npp[i]
        im = nmm[i]
#        i = int(rr[skip + 3*ivol] * nlat)
#        j = int(rr[skip + 3*ivol + 1] * nlat)

        force = field[ip] + field[im]
        phi = field[i]
        rand_num = rr[skip + 2*i]
        phi_prova = phi + 2 * d_metro * (0.5 - rand_num) 

        p_rat = c1 * phi_prova * force - c2 * phi_prova**2
        p_rat = p_rat - c1 * phi * force + c2 * phi**2
        rand_num = log(rr[skip + 2*i + 1])
        if x < p_rat: field[i] = phi_prova
#=============================================================================

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
