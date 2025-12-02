from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension
import numpy

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

triangle_hash_module = Extension(
    'libmesh.triangle_hash',
    sources=[
        'libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

## Gather all extension modules
ext_modules = [triangle_hash_module]

setup(
    ext_modules=cythonize(ext_modules),
    # ext_modules=cythonize([
    #         'libmesh/triangle_hash.pyx',
    #     ]),
    # packages=find_packages(),
    cmdclass={
        'build_ext': BuildExtension
    },
    # name = 'libmesh',
    # version = '0.1.0',
    # author = 'ConvONet'
    name='dsrb',
    # packages=find_packages(include=['dsrb']),
    packages=find_packages(),
    version='0.1.0',
    description='dsr-benchmark',
    author='Raphael Sulzer',
)

# setup(
#     name='dsrb',
#     # packages=find_packages(include=['dsrb']),
#     packages=find_packages(),
#     version='0.1.0',
#     description='dsr-benchmark',
#     author='Raphael Sulzer',
# )