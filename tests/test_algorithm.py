from tdasampling import algorithm
from tdasampling import sampling_setup
import tempfile
from os.path import join
import os


def test_check_for_nonzero_entry():
    assert algorithm.check_for_nonzero_entry([0, 0, 0, 0, 0]) == False
    assert algorithm.check_for_nonzero_entry([0, 0, 0, 0, 1]) == True
    assert algorithm.check_for_nonzero_entry([0, 0, 0, 1, 0]) == True
    assert algorithm.check_for_nonzero_entry([0, 0, 1, 0, 0]) == True
    assert algorithm.check_for_nonzero_entry([0, 1, 0, 0, 0]) == True
    assert algorithm.check_for_nonzero_entry([1, 0, 0, 0, 0]) == True
    assert algorithm.check_for_nonzero_entry([1, 1, 1, 1, 1]) == True


#	parser.add_option("--mpiexecutable",dest="mpi_executable_location",default="mpiexec",help="Use this option to specify a path to mpiexec if it is not in your PATH by default.")
#	parser.add_option("--hosts",dest="hosts",help="Comma separated list of ssh host names to use for MPI functionality of Bertini.")
#	parser.add_option("--bertini",dest="bertini_executable_location",default="bertini",help="Use this option to specify a path to bertini if it is not in your PATH by default.")
#	parser.add_option("--processors",dest="number_of_bertini_processes",type=int,default=1,help="Number of processors to use for solving initial system using Bertini.")


def test_sampling_setup():
    tempdir = tempfile.mkdtemp()
    parameters = {
        "mpi_executable_location": "mpiexec",
        "hosts": None,
        "bertini_executable_location": "bertini",
        "number_of_bertini_processes": 1
    }
    args = [tempdir]
    with open(join(tempdir, "polynomial_system"), "w") as polynomial_system:
        polynomial_system.write("x1,y1\nx1^2 + y1^2 - 1\n")

    sampling_setup(parameters, args)
    assert True == True



