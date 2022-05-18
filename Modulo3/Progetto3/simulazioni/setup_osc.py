from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

module_name = 'oscillatore'

e1 = Extension('oscillatore', ['oscillatore.pyx'], include_dirs=[numpy.get_include(), '.'], extra_compile_args=["-O3"])

ext_modules = [e1]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3

setup(
    name = module_name,
    ext_modules = cythonize(ext_modules, annotate=True)
    )
