#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2019--, Cell2cell development team.
#
# Distributed under the terms of the MIT License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools.command.egg_info import egg_info
from setuptools.command.develop import develop
from setuptools.command.install import install
import re
import ast
import os
from setuptools import find_packages, setup

# Dealing with Cython
USE_CYTHON = os.environ.get('USE_CYTHON', False)
ext = '.pyx' if USE_CYTHON else '.c'


def custom_command():
    import sys
    if sys.platform in ['darwin', 'linux']:
        os.system('pip install numpy')

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        custom_command()

class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        custom_command()

class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        custom_command()


extensions = [
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

classes = """
    Development Status :: 2 - Pre-Alpha
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = ('TBD')

with open('README.md') as f:
    long_description = f.read()

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('cell2cell/__init__.py', 'rb') as f:
    hit = _version_re.search(f.read().decode('utf-8')).group(1)
    version = str(ast.literal_eval(hit))

setup(name='cell2cell',
      version=version,
      license='BSD-3-Clause',
      description=description,
      long_description=long_description,
      author="cell2cell development team",
      author_email="earmingol@eng.ucsd.edu",
      maintainer="cell2cell development team",
      maintainer_email="earmingol@eng.ucsd.edu",
      packages=find_packages(),
      ext_modules=extensions,
      install_requires=['numpy >= 1.16',
                        'pandas >= 1.0.0',
                        'xlrd >= 1.1',
                        'openpyxl >= 2.6.2',
                        'networkx >= 2.3',
                        'matplotlib >= 3.1.2',
                        'seaborn >= 0.9.0',
                        'goenrich >= 1.11',
                        'scikit-bio',
                        'scikit-learn',
                        'tqdm'
                        ],
      classifiers=classifiers,
      entry_points={},
      package_data={},
      cmdclass={'install': CustomInstallCommand,
                'develop': CustomDevelopCommand,
                'egg_info': CustomEggInfoCommand, },
      zip_safe=False)