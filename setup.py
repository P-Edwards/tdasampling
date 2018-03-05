from setuptools import setup

setup(name="tdasampling", 
	version="1.0.0", 
	description="Compute dense samples of real algebraic varieties for use with topological data analysis tools.", 
	url="https://github.com/P-Edwards/tdasampling",
	author="Parker Edwards",
	author_email="pedwards@ufl.edu", 
	license="MIT",
	packages=["tdasampling"],
	classifiers =["Development Status :: 3 - Alpha","License :: OSI Approved :: MIT License","Programming Language :: Python :: 2.7"],
	install_requires=["numpy","rtree","sympy"], 
	scripts=["bin/tdasampling","bin/sampling-setup"],
	depedency_links=["https://bertini.nd.edu/"],
	zip_safe=False)