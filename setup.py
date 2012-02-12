from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension(
                   "PyKf",                 # name of extension
                   ["PyKf.pyx", "kf.cpp"], #  our Cython source
                   language="c++")],  # causes Cython to create C++ source
      cmdclass={'build_ext': build_ext})
