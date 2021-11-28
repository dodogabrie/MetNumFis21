from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

file_name = 0
for file in os.listdir("."):
    if file.endswith(".pyx"):
        file_name = file
if file_name == 0:
    raise ValueError('File pyx not found!')

module_name = file_name.split('.')[0]

print(module_name)
setup(
    ext_modules=cythonize(
        Extension(
            module_name, [file_name],
            extra_compile_args=["-ffast-math"],
            include_dirs=[numpy.get_include()],
            compiler_directives={'language_level' : "3"}
        )
    )    
)
