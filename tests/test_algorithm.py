from tdasampling import algorithm
from tdasampling import sampling_setup
from tdasampling import sampling
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
    assert os.path.exists(join(tempdir, "minimizer/failed_paths"))
    assert os.path.exists(join(tempdir, "minimizer/finite_solutions"))
    assert os.path.exists(join(tempdir, "minimizer/input_param"))
    assert os.path.exists(join(tempdir, "minimizer/input_start"))
    assert os.path.exists(join(tempdir, "minimizer/main_data"))


def test_sampling():
    tempdir = tempfile.mkdtemp()
    parameters = {
        "mpi_executable_location": "mpiexec",
        "hosts": None,
        "bertini_executable_location": "bertini",
        "number_of_bertini_processes": 1
    }
    args = [tempdir]
    with open(join(tempdir, "polynomial_system"), "w") as polynomial_system:
        polynomial_system.write("x1,y1\n(x1-1)^2 + (y1-1)^2 - 1\n")

    sampling_setup(parameters, args)

    parameters = {
        "number_of_parallel_instances": 1,
        "number_of_processors_for_bertini": 1,
        "parameter_file": False,
        "mpi_executable_location": None,
        "bertini_executable_location": None,
        "total_number_of_processors": None,
        "rounding_precision": 1e-7,
        "output_file_name": None,
        "dimensionality": None,
        "previous_points": None,
        "old_density": None,
        "hosts": None,
        "skip_interval": None,
        "rolling_average_length": None,
        "number_of_allowed_skips": None,
        "output_path": None,
        "variable_indices": None
    }
    args = ["0,2,0,2", "1", "1", tempdir]
    sampling(parameters, args)

    print(os.listdir(tempdir))
    assert os.path.exists(join(tempdir, "100e-2_sample.txt"))


