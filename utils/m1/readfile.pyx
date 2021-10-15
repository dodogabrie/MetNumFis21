""" This Module read a 2 COLUMNS file of data in a fast way """
from libc.stdio cimport *
from libc.stdlib cimport atof
from libc.string cimport strtok
import numpy as np
cimport cython, numpy as np # Make numpy work with cython

# C definitions ###############################################
ctypedef np.double_t DTYPE_t
cdef extern from "stdio.h":
    FILE *fopen(const char *, const char *)
    int fclose(FILE *)
    ssize_t getline(char **, size_t *, FILE *)
############################################################### 

# Main function ###############################################
def fastload(filename, int Ndata):
    """
    Function for fast extraction of data from txt file.
    NOTE: do not import matplotlib before importing this module, in that 
    case the module will not work...

    Parameters
    ----------
    filename : 'b'string
        String containing the file .txt and his path preceeded by the letter b.
        For example b'mydata.txt'
    Ndata : int
        Maximum number of data contained in the file.

    Return: 2d numpy array
        Array of x/y containing the data in the txt columns.
    """
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] data = np.empty((Ndata, 2))
    filename_byte_string = filename
    cdef char* fname = filename_byte_string
    cdef FILE* cfile
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)
    cdef int i = take_data(data, cfile) 
    return data[:i]
###############################################################

# Core function (do the hard word) ############################
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int take_data(DTYPE_t[:,:] data, FILE* cfile):
    cdef int i = 0
    cdef char * line = NULL
    cdef char * line1 = NULL
    cdef char * line2 = NULL
    cdef size_t l = 0
    cdef ssize_t read
    while True:
        read = getline(&line, &l, cfile)
        if read == -1: break
        line1 = strtok(line, " ")
        line2 = strtok(NULL, " ")
        line2 = strtok(line2, "\n")
        data[i][0] = atof(line1)
        data[i][1] = atof(line2)
        i+=1
    fclose(cfile)
    return i 
###############################################################

# Main function for slow load data ############################
def slowload(filename, int Ndata):
    """
    Function for slow extraction of data from txt file.
    Just use the 'open' native python function and string operations.

    Parameters
    ----------
    filename : 'b'string
        String containing the file .txt and his path preceeded by the letter b.
        For example b'mydata.txt'
    Ndata : int
        Maximum number of data contained in the file.

    Return: 2d numpy array
        Array of x/y containing the data in the txt columns.
    """

    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] data = np.empty((Ndata, 2))
    cdef int i = 0
    with open(filename, 'rb') as f:
        for line in f:
            data[i] = line.strip().split()
            i+=1
    return data[:i]
###############################################################
