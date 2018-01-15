from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("GF_fiber_cython.pyx"),
)

# execute with
# python setup.py build_ext --inplace