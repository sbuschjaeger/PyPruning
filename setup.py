import multiprocessing
import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion

from setuptools import setup

setup(
    name='PyPruning',
    version='0.1',
    author='Sebastian Buschjäger',
    author_email='sebastian.buschjaeger@tu-dortmund.de',
    description='Prune ensembles in Python',
    long_description='Prune ensembles in Python',
    zip_safe=False,
    install_requires = [
        "numpy",
        "scikit-learn",
        "pip",
        "setuptools",
        "tqdm",
        "cvxpy"
    ]
)
