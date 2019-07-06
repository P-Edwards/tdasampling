tdasampling
-------------

Copyright (C) 2019 `Parker
Edwards <https://people.clas.ufl.edu/pedwards>`__

External requirements
---------------------

1. `Bertini <https://bertini.nd.edu/>`__
2. `Libspatialindex <https://libspatialindex.github.io/>`__ (A
   requirement of the Python package
   `Rtree <https://pypi.python.org/pypi/Rtree/>`__)
3. An mpiexec executable, like the one you can compile from source
   `here <https://www.open-mpi.org/software/ompi/v3.0/>`__

Description
-----------

Python package for sampling real algebraic varieties from their
polynomial systems. See the article https://arxiv.org/abs/1802.07716 for
theoretical details. It has been tested on Linux.

The package installs two command line scripts:

1. **tdasampling** - Entry point to the main sampling algorithm.
2. **sampling-setup** - Script for setting up a directory for sampling
   computation from just a list of polynomials in the system.

See the included tutorial for detailed information about all the
different options.

Version 1.1.3
-------------

Basic usage for tdasampling
---------------------------

.. code:: shell

    $ tdasampling [options] bounds density number_of_functions_in_system execution_directory

-  Bounds is a list of a form like -1.0,1.0,-1.0,1.0, which indicates
   the region in which to sample the polynomial system is box [-1.0,1.0]
   x [-1.0,1.0] in Euclidean space
-  execution\_directory is a directory containing, at minimum:
-  A *minimizer* directory which contains parameter homotopy files for
   Bertini. Unless you have experience with Bertini, set these up with
   ``sampling-setup``
-  (*Recommended, not required*) A parameters file *parameters.json*.
   See examples for format. If you include a *parameters.json* file and
   use the option flag ``--parameters`` with ``tdasampling``, the
   *parameters.json* file should include all the information except
   ``execution_directory``, which can then be omitted from the command
   line call.

Basic usage for sampling-setup
------------------------------

.. code:: shell

    $ sampling-setup [options] path_to_directory_to_setup

-  The directory indicated at ``path_to_directory_to_setup`` should
   contain a file named *polynomial\_system*. The general format of that
   file is text:

::

    list of variable names separated by commas
    polynomial 1
    polynomial 2 
    ...
    polynomial n

For example, if we were sampling from a circle of radius 1:

::

    x1,x2
    x1^2 + x2^2 - 1

-  ``--mpiexecutable /a/path/to/mpiexec`` option to indicate a path to
   ``mpiexec``. Unnecesssary if your ``mpiexec`` can be called as
   ``mpiexec``
-  ``--bertini /a/path/to/bertini``: a path to your ``bertini``
   executable if it cannot be called as ``bertini``
-  ``--processors k``: the number of processes you would like to use for
   the ``bertini`` solving run associated with setup
-  ``--hosts name1,name2,...,namek``: list of ssh names for nodes to use
   for the ``bertini`` computation. By default, the ``bertini`` run will
   run only on your local machine

License
-------

tdasampling is licensed under an MIT license.
