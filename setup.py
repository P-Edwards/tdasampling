from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here,"README.rst"),"r") as readmefile: 
	readme = readmefile.read()

setup(name="tdasampling", 
	version="1.0.0", 
	description="Compute dense samples of real algebraic varieties for use with topological data analysis tools.", 
	long_description=readme,
	url="https://github.com/P-Edwards/tdasampling",
	author="Parker Edwards",
	author_email="pedwards@ufl.edu", 
	license="MIT",
	packages=["tdasampling"],
	classifiers =["Development Status :: 3 - Alpha","License :: OSI Approved :: MIT License","Programming Language :: Python :: 2.7"],
	install_requires=["numpy","rtree","sympy"], 
	scripts=["bin/tdasampling","bin/sampling-setup"],
	dependency_links=["https://bertini.nd.edu/"],
	zip_safe=False)