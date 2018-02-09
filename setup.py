# -*- coding: utf-8 -*-

# Learn more: https://github.com/soumik-kanad/FeatSel/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='FeatSel',
    version='0.1.0',
    description='',
    long_description=readme,
    author='Rishav Chourasia,',
    author_email='rishav.chourasia@gmail.com, ',
    url='https://github.com/soumik-kanad/FeatSel',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'datasamples'))
    requires=['pytest', 'spinx', 'sklearn', 'numpy', 'pandas']
)
