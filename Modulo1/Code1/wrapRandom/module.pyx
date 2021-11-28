# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
import time
import numpy as np
cimport cython, numpy as np # Make numpy work with cython
from libc.math cimport sin  # How to import C function

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"


def fname(int N):
    """
    Testing random number generator from Cpp wrapped againist the 
    generator provided by numpy.
    """
    loop_random(N)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Make division fast like C
cdef void loop_random(int N):
    cdef mt19937 gen = mt19937(int(time.time()))
    cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
    cdef int i
    for i in range(N):
        dist(gen)
        dist(gen)
