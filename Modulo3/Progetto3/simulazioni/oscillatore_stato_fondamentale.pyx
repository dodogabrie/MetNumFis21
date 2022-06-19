# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
import numpy as np
import os
import math
cimport numpy as np # Make numpy work with cython
cimport cython
from libc.math cimport exp, log
from libc.stdio cimport printf
from libc.time cimport time,time_t
import json

#_______ Define the random number generator _______________________________________________
# Algorithm: Mersenne Twister 19937  [ https://en.wikipedia.org/wiki/Mersenne_Twister ]
cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937: # Import the class from the C++ library
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed
    
    cdef cppclass uniform_real_distribution[T]: # Use the Mersenne Twister 
                                                # to extract from uniform distribution
        uniform_real_distribution() #Import the algorithm from the C++ library
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"

# define the random generator (globally)
cdef:
    mt19937 gen
    uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
#___________________________________________________________________________________________
 

def simulator(int nlat, int iflag, 
              int measures, int i_decorrel, int i_term, double d_metro,
              double eta, int seed = -1, str data_dir = "", file_name = 'lattice'):
    """
    Main function for the harmonic oscillator.
    Parameters
    ----------
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
    seed: int
        The seed for the random number generator.
    Results
    -------
    """
    global gen, dist # Use the global "variables" from the random number extraction
    if seed == -1: seed = int(time(NULL)) # If seed is not given, use the current time
    gen = mt19937(seed) # Initialize the random number generator

    # constant for metropolis
    cdef double c1 = 1/eta
    cdef double c2 = c1 + eta/2
    cdef double inv_nlat = 1./nlat

    # geometry array
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] npp = np.zeros(nlat).astype(int)
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] nmm = np.zeros(nlat).astype(int)
    # field array
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] field = np.ones(nlat)

    cdef int i, idec # index for loops

    # Index for counting the remaining time
    cdef int count = 0, perc_count = 0, count_max = int(i_term/10)
    cdef time_t t0 = time(NULL), sum_t = 0, frac_elapsed = 0

    # 0) Initialize the lattice
    geometry(nlat, npp, nmm) # Set the boundary conditions
    inizialize_lattice(iflag, nlat, field) # Inizialize the lattice

    lattice_dir = f'../dati/' + data_dir + '/'       # Directory where the lattice will be saved
    lattice_name_file = file_name + '_lattice'

    if not os.path.exists(os.path.dirname(lattice_dir)): # If the directory does not exist
        print('folder ', lattice_dir, ' does not exist. Creating...')
        os.makedirs(os.path.dirname(lattice_dir)) # Create the directory

    data_dict = {'seed': seed, 'eta':eta, 'nlat': nlat, 
                 'iflag': iflag, 'measures': measures,
                 'i_decorrel': i_decorrel, 'i_term': i_term, 
                 'd_metro': d_metro}
    with open(lattice_dir + lattice_name_file + f'.json', 'w') as f: # Save the parameters in a .json file
        json.dump(data_dict, f) # the .json file contains the dictionary

    # 1) Termalization step
    print("Termalization step")
    for i in range(i_term):
        count, sum_t, perc_count, t0 = print_counter(count, perc_count, count_max, t0, 
                                                     frac_elapsed, sum_t)
        update_metropolis(field, nlat, d_metro, npp, nmm, eta, c1, c2) # MC
    print("Termalization step finished")

    count = 0
    perc_count = 0
    count_max = int(measures/10)
    t0 = time(NULL)
    sum_t = 0
    frac_elapsed = 0


    # 2) Measures step
    for i in range(measures):
        # a) Print counter and time remaining
        count, sum_t, perc_count, t0 = print_counter(count, perc_count, count_max, t0, 
                                                     frac_elapsed, sum_t)
        # c) Decorrelate the lattice
        for idec in range(i_decorrel):
            update_metropolis(field, nlat, d_metro, npp, nmm, eta, c1, c2) # MC

        # d) Measure the observable
        # 3) Save the data
        np.savetxt(lattice_dir + lattice_name_file + f'{i}.dat', field) # Save the lattice
    print(f"Done! nlat {nlat}, eta {eta}")
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
cdef void inizialize_lattice(int iflag, int nlat, np.double_t[:] field, ):
    """
    Inizialize lattice values according to the iflag.
    """
    global gen, dist
    cdef double x
    cdef int i

    if iflag == 0: # Cold start --> all site 0
        for i in range(nlat):
            field[i] = 0.
    else:
        if iflag == 1: # Warm start --> Random y
            for i in range(nlat):
                y = 1 - 2*dist(gen) # Random number in (-1, 1)
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
                                   double eta, double c1, double c2): # simulation parameters
    """
    Update the lattice with a metropolis.
    """
    global gen, dist
    cdef int i, ip, im
    cdef double force, p_rat, phi, phi_prova
    cdef double rand_num
    for i in range(nlat):
        ip = npp[i]
        im = nmm[i]

        force = field[ip] + field[im]
        phi = field[i]
        rand_num = dist(gen)
        phi_prova = phi + 2 * d_metro * (0.5 - rand_num) 

        p_rat = c1 * phi_prova * force - c2 * phi_prova**2
        p_rat = p_rat - c1 * phi * force + c2 * phi**2
        rand_num = log(dist(gen))
        if rand_num < p_rat: field[i] = phi_prova
#=============================================================================
