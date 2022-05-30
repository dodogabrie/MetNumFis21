""" This Module read a 2 COLUMNS file of data in a fast way """
from libc.stdio cimport *
from cython.parallel import prange
from libc.stdlib cimport atof, free, malloc
from libc.string cimport strtok, strncmp, strlen, strcpy
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
def fastload(filename, comments = '#', delimiter = ' ', usecols = None):
    """
    Function for fast extraction of data from txt file.
    NOTE: do not import matplotlib before importing this module, in that 
    case the module will not work...

    Parameters
    ----------
    filename : 'b'string
        String containing the file .txt and his path preceeded by the letter b.
        For example b'mydata.txt'

    Return: 2d numpy array
        Array of x/y containing the data in the txt columns.
    """
    cdef int i
    #cdef int len_comment = len(comments), len_delimiter = len(delimiter), len_fname = len(filename)

    cdef char * cdelimiter = <char*>malloc(10 * sizeof(char))
    cdef char * ccomments  = <char*>malloc(10 * sizeof(char))
    cdef char * fname      = <char*>malloc(100 * sizeof(char))
    strcpy(cdelimiter, delimiter.encode('utf-8'))
    strcpy(ccomments, comments.encode('utf-8'))
    strcpy(fname, filename.encode('utf-8'))

    Ndata, Ncol = get_dimension(fname, cdelimiter, ccomments)

    # usecols handling
    cdef np.ndarray cols = np.arange(Ncol).astype(int)
    cdef int max_col 
    cdef int num_col = Ncol
    cdef int check_cols = 1
    if usecols is not None:
        cols = np.array(usecols).astype(int)
        max_col = np.max(usecols)
        num_col = len(usecols) # eventually overwrite the number of columns
        if Ncol <= max_col:
            check_cols = 0

    cdef np.ndarray data = np.empty((Ndata, num_col)).astype(np.double)

    if check_cols:
        take_data(data, fname, Ndata, Ncol, cdelimiter, ccomments, cols) 
    else:
        printf("ERROR: index of columns out of range, you asked %d but the file contains only %d columns\n", max_col, Ncol)
    return np.squeeze(data)
###############################################################


cdef (int, int) get_dimension(char * fname, char * delimiter, char * comments):
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % fname)
    cdef int Ndata=0, Ncol=0
    cdef char * line = NULL
    cdef size_t l = 0
    cdef ssize_t read
    while True:
        read = getline(&line, &l, cfile)
        if read == -1: break
        if Ndata == 0:
            line = strtok(line, delimiter)
            if StartsWith(line, comments): continue
            else:
                while line != NULL:
                    line = strtok(NULL, delimiter)
                    Ncol += 1
        Ndata += 1
    free(line)
    fclose(cfile)
    return Ndata, Ncol

# Core function (do the hard word) ############################
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void take_data(DTYPE_t[:,:] data, char * fname, int Ndata, int Ncol, 
        char * delimiter, char * comments, np.int_t[:] cols):
    cfile = fopen(fname, "rb")
    cdef int i, j = 0, j_max = 0
    cdef int col_counter = 0 # column index
    cdef int c = cols[col_counter]
    cdef char * line = NULL
    cdef char * token 
    cdef size_t l = 0
    cdef ssize_t read = 0
    for i in range(Ndata):
        read = getline(&line, &l, cfile)
        if read == -1: break
        token = strtok(line, delimiter)
        if StartsWith(line, comments): continue
        for j in range(Ncol):
            if j == c:
                data[i][col_counter] = atof(token)
                col_counter += 1
                c = cols[col_counter]
            token = strtok(NULL, delimiter)
        col_counter = 0
        c = cols[col_counter]
    free(line)
    fclose(cfile)
###############################################################

def get_data_shape(filename, comments = '#', delimiter = ' '):
    """
    Function for fast extraction of data from txt file.
    NOTE: do not import matplotlib before importing this module, in that 
    case the module will not work...

    Parameters
    ----------
    filename : 'b'string
        String containing the file .txt and his path preceeded by the letter b.
        For example b'mydata.txt'

    Return: 2d numpy array
        Array of x/y containing the data in the txt columns.
    """
    def turn_utf8(data):
        "Returns an utf-8 object on success, or None on failure"
        try: # Try to encode the data as utf-8
            return data.encode('utf-8')
        except AttributeError: # if already utf-8
            return data

    cdef bytes cdelimiter = turn_utf8(delimiter)
    cdef bytes ccomments = turn_utf8(comments)
    cdef bytes fname = turn_utf8(filename)

    Ndata, Ncol = get_dimension(fname, cdelimiter, ccomments)
    return Ndata, Ncol
#

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
