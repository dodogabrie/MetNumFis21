import numpy as np
import os
import math
cimport numpy as np # Make numpy work with cython
cimport cython
from libc.math cimport exp, log
from libc.stdio cimport printf
from libc.time cimport time,time_t
import json

def simulator(int seed, int nlat, int iflag, 
              int measures, int i_decorrel, int i_term, double d_metro,
              double eta, int save_data = 1, int save_lattice = 0):
    """
    Main function for the harmonic oscillator.
    Parameters
    ----------
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
    """
    # define the random generator
    rng =  np.random.Generator(np.random.PCG64())
    # geometry array
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] npp = np.zeros(nlat).astype(int)
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] nmm = np.zeros(nlat).astype(int)
    # results array
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] obs1_array = np.empty(measures)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] obs2_array = np.empty(measures)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] single_obs = np.empty(2)
    # random array
    cdef int n_rand_metro = 2 # Number of random numbers for the single metropolis
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] rr = np.empty(n_rand_metro*i_decorrel*nlat)
    cdef int count_term_max = int(1e5)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] r_term = np.empty(n_rand_metro*count_term_max*nlat)
    # lattice array
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] field = np.ones(nlat)

    cdef int i, idec # index for loops

    # Index for counting the remaining time
    cdef int count = 0, perc_count = 0, count_max = int(measures/10)
    cdef time_t t0 = time(NULL), sum_t = 0, frac_elapsed

    # 0) Initialize the lattice
    geometry(nlat, npp, nmm) # Set the boundary conditions
    inizialize_lattice(iflag, nlat, field) # Inizialize the lattice

    # 1) Termalization step
    #r_term = rng.uniform(size = n_rand_metro*i_term*nlat) # Extract the random points for the MC
    cdef int count_term = 0
    r_term = rng.uniform(size = n_rand_metro*count_term_max*nlat) 
    print("Termalization step")
    for i in range(i_term):
        update_metropolis(field, nlat, d_metro, npp, nmm, eta, r_term, count_term*n_rand_metro*nlat) # MC
        count_term+=1
        if count_term == count_term_max:
            count_term = 0
            r_term = rng.uniform(size = n_rand_metro*count_term_max*nlat) 
    print("Termalization step finished")
    # 2) Measures step
    for i in range(measures):
        # a) Print counter and time remaining
        count, sum_t, perc_count, t0 = print_counter(count, perc_count, count_max, t0, 
                                                     frac_elapsed, sum_t)
        # b) Extract the random points for the MC
        rr = rng.uniform(size = n_rand_metro*i_decorrel*nlat)

        # c) Decorrelate the lattice
        for idec in range(i_decorrel):
            update_metropolis(field, nlat, d_metro, npp, nmm, eta, rr, idec*n_rand_metro*nlat)

        # d) Measure the observables
        get_measures(nlat, field, npp, single_obs)
        obs1_array[i] = single_obs[0]
        obs2_array[i] = single_obs[1]

    # 3) Save the data
    store_results( seed, nlat, iflag, measures, i_decorrel, i_term, d_metro,
                   eta, save_data, save_lattice, obs1_array, obs2_array, field)
#==============================================================================

#=============== FUNCTION TO STORE THE RESULTS IN FILES =======================
def store_results(seed, nlat, iflag, measures, i_decorrel, i_term, d_metro,
                  eta, save_data, save_lattice, obs1_array, obs2_array, field):
    """
    Store the results in data files.
    """
    # Dictionary with parameters of simulation
    data_dict = {'seed': seed, 'eta':eta, 'nlat': nlat, 
                 'iflag': iflag, 'measures': measures,
                 'i_decorrel': i_decorrel, 'i_term': i_term, 
                 'd_metro': d_metro}
    if save_data: # If the user want to save the observables
        data_dir = f'../dati/obs_nlat{nlat}/' # Directory where the data will be saved
        if not os.path.exists(os.path.dirname(data_dir)): # If the directory does not exist
            os.makedirs(os.path.dirname(data_dir)) # Create the directory
        # Save the data in a .dat file
        np.savetxt(data_dir + f'data_eta{eta}.dat' , np.column_stack((obs1_array, obs2_array)))
        with open(data_dir + f'data_eta{eta}.json', 'w') as f: # Save the parameters in a .json file
            json.dump(data_dict, f) # a .json file contains a dictionary
    if save_lattice: # If the user want to save the lattice
        lattice_dir = f'../dati/lattice_nlat{nlat}/' # Directory where the lattice will be saved
        if not os.path.exists(os.path.dirname(lattice_dir)): # If the directory does not exist
            os.makedirs(os.path.dirname(lattice_dir)) # Create the directory
        np.savetxt(lattice_dir + f'lattice_eta{eta}.dat', field) # Save the lattice
        with open(lattice_dir + f'data_eta{eta}.json', 'w') as f: # Save the parameters in a .json file
            json.dump(data_dict, f) # the .json file contains the dictionary
    return
#==============================================================================

#=============== FUNCTION TO EVALUATE THE TIME REMAINING ======================
@cython.cdivision(True)
cdef (int, time_t, int, time_t) print_counter(int count, int perc_count, time_t 
                                              count_max, time_t t0,
                                              time_t frac_elapsed, time_t sum_t):
    """
    Print the percentage of the simulation completed
    Parameters
    """
    cdef time_t t1
    count+=1
    if count >= count_max:
        perc_count += 1
        t1 = time(NULL)
        frac_elapsed = t1 - t0
        sum_t += frac_elapsed
        printf("%d / 10 --> %ld s left\n", 
                perc_count, (10 - perc_count)*sum_t/perc_count)
        t0 = t1
        count = 0
    return count, sum_t, perc_count, t0

#=================== FUNCTION TO DEFINE THE GEOMETRY ==========================
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
#==============================================================================

#=================== FUNCTION TO INITIALIZE THE LATTICE =======================
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

#=================== FUNCTION FOR THE METROPOLIS STEP ========================
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Enable cdivision.
cdef inline void update_metropolis(np.double_t[:] field, # the field
                                   int nlat, # lateral size
                                   double d_metro, # delta for the metropolis
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

        force = field[ip] + field[im]
        phi = field[i]
        rand_num = rr[skip + 2*i]
        phi_prova = phi + 2 * d_metro * (0.5 - rand_num) 

        p_rat = c1 * phi_prova * force - c2 * phi_prova**2
        p_rat = p_rat - c1 * phi * force + c2 * phi**2
        rand_num = log(rr[skip + 2*i + 1])
        if rand_num < p_rat: field[i] = phi_prova
#=============================================================================

#=================== FUNCTION FOR THE MEASURE ================================
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline void get_measures(int nlat, np.double_t[:] field, np.int_t[:] npp, np.double_t[:] single_obs):
    cdef double obs1 = 0
    cdef double obs2 = 0
    cdef int i
    for i in range(nlat):
        obs1 = obs1 + field[i]**2                   # media sul singolo path di y^2 
        obs2 = obs2 + (field[i]-field[npp[i]])**2   # media sul singolo path di Delta y^2
    single_obs[0] = obs1/nlat
    single_obs[1] = obs2/nlat
#=============================================================================
