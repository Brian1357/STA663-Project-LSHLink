import setuptools
from setuptools import Extension
from Cython.Build import cythonize, build_ext
import numpy

with open("README.md", "r") as fh:
  long_description = fh.read()
  
setuptools.setup(
    ext_modules = cythonize("LSHlinkCython/LSHlink_Cython.pyx"),
    zip_safe=False,
    include_dirs = [numpy.get_include(), "LSHlinkCython/"])
