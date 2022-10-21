from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

module_name = 'readfile'

e1 = Extension('readfile', ['readfile.pyx'], include_dirs=[numpy.get_include(), '.'],)

ext_modules = [e1]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3

setup(
    name = module_name,
    ext_modules = cythonize(ext_modules, annotate=True)
    )
