#example_cython.pyx
def test(int x):
    cdef int y = 0, i
    for i in range(x):
        y += i
    return y
