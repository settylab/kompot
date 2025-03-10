#!/usr/bin/env python

import os
from setuptools import setup, find_packages

# Get version
exec(open('kompot/version.py').read())

# Read description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='kompot',
    version=__version__,
    description='Differential abundance and gene expression analysis using Mahalanobis distance with JAX backend',
    author='Dominik Otto',
    author_email='dotto@fredhutch.org',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/settylab/kompot',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'jax>=0.3.0',
        'jaxlib>=0.3.0',
        'mellon>=1.5.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'tqdm>=4.60.0',
        'anndata>=0.8.0',
        'pynndescent>=0.5.0',
        'python-igraph>=0.10.0',
        'leidenalg>=0.9.0',
        'importlib-resources>=5.0.0;python_version<"3.9"',
    ],
    extras_require={
        'docs': [
            'sphinx>=7.0.0',
            'nbsphinx>=0.9.0',
            'furo>=2024.0.0',
            'sphinx-autodocgen>=1.0.0',
            'sphinx-github-style>=1.2.0',
            'lxml[html_clean]',
            'IPython',
        ],
        'plot': [
            'scanpy>=1.9.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.12.0',
        ],
        'recommended': [
            'palantir>=1.0.0',
        ],
        'all': [
            'scanpy>=1.9.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.12.0',
            'palantir>=1.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)