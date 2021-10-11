from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(Extension(
            "Mysin", ["Mysin.pyx"],
            extra_compile_args=["-ffast-math"],
            include_dirs=[numpy.get_include()],
            compiler_directives={'language_level' : "3"}
            )
    )    
)
