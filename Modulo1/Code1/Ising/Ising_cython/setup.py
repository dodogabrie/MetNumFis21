from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(Extension(
            "ising", ["ising.pyx"],
            extra_compile_args=["-ffast-math"],
            language="c++",
            include_dirs=[numpy.get_include()]
            )
        )    
    )
