import io
import os
import sys
from setuptools import find_packages, setup

# Package meta-data.
NAME = 'nlpgnn'
DESCRIPTION = 'An out-of-the-box NLP toolkit can easily help you solve tasks such as\
              Entity Recognition, Text Classification, Relation Extraction and so on.'
URL = 'https://github.com/kyzhouhzau/nlpgnn'
EMAIL = 'zhoukaiyinhzau@gmail.com'
AUTHOR = 'Kaiyin Zhou'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.0'

REQUIRED = [
    'typeguard',
    'gensim',
    'tqdm',
    'regex',
    'scikit-learn',
    'sentencepiece',
    'networkx'
]

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    include_package_data=True,
    license='Apache',
    classifiers=[
        'Programming Language :: Python',
    ],
)