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
 

def simulator_f2(input_list_k, int nlat, int iflag, 
                 int measures, int i_decorrel, int i_term, double d_metro,
                 double eta, int save_data = 1, int save_lattice = 0, int seed = -1, 
                 str data_dir = "", file_name = None, int verbose = 10):
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
    cdef int i, idec # index for loops

    # geometry array
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] npp = np.zeros(nlat).astype(int)
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] nmm = np.zeros(nlat).astype(int)
    # results array
    cdef int num_k = len(input_list_k)
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] k_list = np.empty(num_k).astype(int)
    for i in range(num_k): 
        k_list[i] = input_list_k[i]
    print("k_list = ", k_list)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] obs1_array = np.empty((measures, num_k))
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] obs2_array = np.empty((measures, num_k))
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] obs1_array_full = np.empty((measures, num_k))
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] obs2_array_full = np.empty((measures, num_k))

    # field array
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] field = np.ones(nlat)


    # Index for counting the remaining time
    cdef int count = 0, perc_count = 0, count_max = int(i_term/verbose)
    cdef time_t t0 = time(NULL), sum_t = 0, frac_elapsed = 0

    # 0) Initialize the lattice
    geometry(nlat, npp, nmm) # Set the boundary conditions
    inizialize_lattice(iflag, nlat, field) # Inizialize the lattice

    # 1) Termalization step
    print("Termalization step")
    for i in range(i_term):
        count, sum_t, perc_count, t0 = print_counter(count, perc_count, count_max, t0, 
                                                     frac_elapsed, sum_t, verbose)
        update_metropolis(field, nlat, d_metro, npp, nmm, eta, c1, c2) # MC
    print("Termalization step finished")

    count = 0
    perc_count = 0
    count_max = int(measures/verbose)
    t0 = time(NULL)
    sum_t = 0
    frac_elapsed = 0


    # 2) Measures step
    for i in range(measures):
        # a) Print counter and time remaining
        count, sum_t, perc_count, t0 = print_counter(count, perc_count, count_max, t0, 
                                                     frac_elapsed, sum_t, verbose)
        # c) Decorrelate the lattice
        for idec in range(i_decorrel):
            update_metropolis(field, nlat, d_metro, npp, nmm, eta, c1, c2) # MC

        # d) Measure the observable
        # !!!!
        for k in range(num_k):
            get_measures(k, i, nlat, inv_nlat, field, npp, k_list, obs1_array, obs2_array, obs1_array_full, obs2_array_full)
        # !!!!

    # 3) Save the data
    store_results( seed, nlat, iflag, measures, i_decorrel, i_term, d_metro,
                   eta, save_data, save_lattice, k_list, obs1_array, obs2_array, 
                   obs1_array_full, obs2_array_full, field, data_dir, file_name)
    print(f"Done! nlat {nlat}, eta {eta}")
#==============================================================================

#=============== FUNCTION TO STORE THE RESULTS IN FILES =======================
def store_results(seed, nlat, iflag, measures, i_decorrel, i_term, d_metro,
                  eta, save_data, save_lattice, k_list, obs1_array, obs2_array,
                  obs1_array_full, obs2_array_full, field, 
                  usr_data_dir, usr_name_file = None):
    """
    Store the results in data files.
    """
    if save_data: print('Saving data...')
    if save_lattice: print('Saving lattice...')
    # Dictionary with parameters of simulation
    data_dict = {'seed': seed, 'eta':eta, 'nlat': nlat, 
                 'iflag': iflag, 'measures': measures,
                 'i_decorrel': i_decorrel, 'i_term': i_term, 
                 'd_metro': d_metro}
    if usr_name_file == None:
        name_file = f'data_eta{eta}_nlat{nlat}'
    else:
        name_file = usr_name_file

    if save_data: # If the user want to save the observables
        data_dir = '../dati/'+ usr_data_dir + '/'         # Directory where the data will be saved
        if not os.path.exists(os.path.dirname(data_dir)): # If the directory does not exist
            os.makedirs(os.path.dirname(data_dir)) # Create the directory
        # Save the data in a .dat file
        np.savetxt(data_dir + name_file + '_Gap_energy_obs1.dat' , np.vstack((k_list,obs1_array)) )
        np.savetxt(data_dir + name_file + '_Gap_energy_obs2.dat' , np.vstack((k_list,obs2_array)) )
        np.savetxt(data_dir + name_file + '_Gap_energy_obs1_full.dat' , np.vstack((k_list,obs1_array_full)) )
        np.savetxt(data_dir + name_file + '_Gap_energy_obs2_full.dat' , np.vstack((k_list,obs2_array_full)) )

        with open(data_dir + name_file + '_Gap_energy.json', 'w') as f: # Save the parameters in a .json file
            json.dump(data_dict, f) # a .json file contains a dictionary
    if save_lattice: # If the user want to save the lattice
        lattice_dir = f'../dati/' + usr_data_dir + '/'       # Directory where the lattice will be saved
        if not os.path.exists(os.path.dirname(lattice_dir)): # If the directory does not exist
            os.makedirs(os.path.dirname(lattice_dir)) # Create the directory
        np.savetxt(lattice_dir + name_file + 'lattice_Gap_energy.dat', field) # Save the lattice
        with open(lattice_dir + name_file + 'lattice_Gap_energy.json', 'w') as f: # Save the parameters in a .json file
            json.dump(data_dict, f) # the .json file contains the dictionary
    return
#==============================================================================

#=============== FUNCTION TO EVALUATE THE TIME REMAINING ======================
@cython.cdivision(True)
cdef (int, time_t, int, time_t) print_counter(int count, int perc_count, time_t 
                                              count_max, time_t t0,
                                              time_t frac_elapsed, time_t sum_t, int verbose):
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
        printf("%d / %d --> %ld s left\n", 
                perc_count, verbose, (verbose - perc_count)*sum_t/perc_count)
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

#=================== FUNCTION FOR THE MEASURE ================================
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline void get_measures(int k, int i, int nlat, double inv_nlat, np.double_t[:] field, 
        np.int_t[:] npp, np.int_t[:] k_list, np.double_t[:,:] obs1_array, 
        np.double_t[:,:] obs2_array, np.double_t[:,:] obs1_array_full, np.double_t[:,:] obs2_array_full):
    # obs1: <y_(j+k) * y_j>_c
    # obs2: <y_(j+k)**2 * y_j**2>_c
    cdef double obs1_sconnessa = 0
    cdef double obs2_sconnessa = 0
    cdef double obs1_connessa = 0
    cdef double obs2_connessa = 0
    cdef int j, kk = k_list[k], jpk
    cdef double f, fk
    for j in range(nlat):
        f = field[j]
        jpk = j + kk
        if jpk >= nlat:
            jpk = jpk - nlat
        fk = field[jpk]
        obs1_sconnessa += fk * f                # aggiungo al totale y_(j+k) * y_j
        obs2_sconnessa += (fk*fk) * (f*f)       # aggiungo al totale y_(j+k)^2 * y_j^2
        obs1_connessa += f 
        obs2_connessa += f * f

    # <O>**2
    obs1_connessa = obs1_connessa*inv_nlat      # divido per N
    obs1_connessa = obs1_connessa * obs1_connessa  # elevo al quadrado
    # ... lo stesso per l'altra
    obs2_connessa = obs2_connessa*inv_nlat
    obs2_connessa = obs2_connessa * obs2_connessa

    # divido per N anche le sconnesse (per mediare)
    obs1_sconnessa = obs1_sconnessa*inv_nlat
    obs2_sconnessa = obs2_sconnessa*inv_nlat

    obs1_array[i, k] = obs1_sconnessa 
    obs1_array_full[i, k] = obs1_sconnessa - obs1_connessa
    obs2_array[i, k] = obs2_sconnessa 
    obs2_array_full[i, k] = obs2_sconnessa - obs2_connessa
#=============================================================================
