#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['hydra-core>=1.3', 'anndata>=0.8.0', 'pytorch_lightning>=1.26.11', 'scanpy', 'scvi-tools']

test_requirements = ['pytest>=3', ]

setup(
    author="Nicholas Sofroniew",
    author_email='sofroniewn@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Modeling cell type specific gene expression",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rosa',
    name='rosa',
    packages=find_packages(include=['rosa', 'rosa.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/sofroniewn/rosa',
    version='0.0.0',
    zip_safe=False,
)
