"""Setup script"""

from distutils.core import setup
from setuptools import find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='PyPSL',
    version='0.0.1',
    author='Bruno Godefroy',
    description='A new library for building PSL models in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['test']),
    classifiers=(
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ),
    python_requires='>=3.6.5',
    install_requires=[
        'joblib',
        'matplotlib',
        'natsort',
        'numpy',
        'packaging',
        'pandas',
        'requests',
        'wget'
    ]
)
