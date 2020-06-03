from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = []
ext_modules.append(
    Extension('trapezion', ['trapezion.pyx'], include_dirs=[np.get_include()])
)


setup(
    name='trapezion',
    version='0.1',
    py_modules=['trapezion'],
    ext_modules = cythonize(ext_modules, include_path=[np.get_include()], language_level=3.7),
    include_dirs=[np.get_include()],
)
