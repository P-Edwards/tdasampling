from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here,"README.rst"),"r") as readmefile: 
	readme = readmefile.read()

setup(name="tdasampling", 
	version="2.0.0", 
	description="Compute dense samples of real algebraic varieties for use with topological data analysis tools.", 
	long_description=readme,
	url="https://github.com/P-Edwards/tdasampling",
	author="Parker Edwards",
	author_email="edwardsp@fau.edu", 
	license="MIT",
	packages=["tdasampling","tdasampling.search_space"],
	classifiers =["Development Status :: 3 - Alpha","License :: OSI Approved :: MIT License","Programming Language :: Python :: >3.8"],
	install_requires=["numpy","Rtree","sympy"], 
        python_requires=">3.8",
	scripts=["bin/tdasampling","bin/sampling-setup"],
	dependency_links=["https://bertini.nd.edu/"],
	zip_safe=False)
