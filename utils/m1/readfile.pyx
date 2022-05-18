""" This Module read a 2 COLUMNS file of data in a fast way """
from libc.stdio cimport *
from libc.stdlib cimport atof
from libc.string cimport strtok, strncmp, strlen
import numpy as np
cimport cython, numpy as np # Make numpy work with cython

# C definitions ###############################################
ctypedef np.double_t DTYPE_t
cdef extern from "stdio.h":
    FILE *fopen(const char *, const char *)
    int fclose(FILE *)
    ssize_t getline(char **, size_t *, FILE *)
############################################################### 
cdef int StartsWith(const char *a, const char *b):
    if strncmp(a, b, strlen(b)) == 0: return 1
    return 0;



# Main function ###############################################
def fastload(filename):
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
    def turn_utf8(data):
        "Returns an utf-8 object on success, or None on failure"
        try: # Try to encode the data as utf-8
            return data.encode('utf-8')
        except AttributeError: # if already utf-8
            return data
    filename_byte_string = turn_utf8(filename)
    cdef char* fname = filename_byte_string
    cdef FILE* cfile
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)

    if Ndata == 0 and Ncol == 0:
        Ndata, Ncol = get_dimension(cfile)
        cfile = fopen(fname, "rb")

    cdef np.ndarray data = np.empty((Ndata, Ncol)).astype(np.double)
    cdef int i, j
    i, j = take_data(data, cfile, Ndata, Ncol) 
    if i != Ndata: print(f"WARNING: the file contain {i} data but you asked for {Ndata}")
    if j != Ncol: print(f"WARNING: the file contain {j} columns but you asked for {Ncol}")
    return data[:i]
###############################################################


cdef (int, int) get_dimension(FILE* cfile):
    cdef int Ndata=0, Ncol=0
    cdef char * line = NULL
    cdef size_t l = 0
    cdef ssize_t read
    while True:
        read = getline(&line, &l, cfile)
        if read == -1: break
#        line1 = strtok(line, " ")
        if Ndata == 0:
            line = strtok(line, " ")
            if StartsWith(line, "#"): continue
            else:
                while line != NULL:
                    line = strtok(NULL, " ")
                    Ncol += 1
        Ndata += 1
    fclose(cfile)
    return Ndata, Ncol

# Core function (do the hard word) ############################
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (int, int) take_data(DTYPE_t[:,:] data, FILE* cfile, int Ndata, int Ncol):
    cdef int i = 0, j = 0, j_max = 0
    cdef char * line = NULL
    cdef size_t l = 0
    cdef ssize_t read
    while True:
        read = getline(&line, &l, cfile)
        if read == -1: break
        line = strtok(line, " ")
        if StartsWith(line, "#"): continue
        while line != NULL:
            if i == 0: j_max += 1
            if i < Ndata:
                if j < Ncol:
                    data[i][j] = atof(line)
                    j += 1
            else:
                break
            line = strtok(NULL, " ")
        i += 1
        j = 0
    fclose(cfile)
    return i, j_max
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
