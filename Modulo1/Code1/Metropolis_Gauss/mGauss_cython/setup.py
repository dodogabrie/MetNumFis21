from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("m_gauss.pyx"),
    include_dirs=[numpy.get_include()]
)    

