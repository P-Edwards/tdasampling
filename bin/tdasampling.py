from tdasampling import sampling_algorithm
from search_space import _splitBoxAlongDimension
from multiprocessing import Process,Manager
import math,sys,os
import numpy as np
import json
from optparse import OptionParser
np.set_printoptions(threshold=np.nan)

usage = ("Usage: %prog [options] bounds density number_of_functions_in_system "
 "polynomial_system_directory\n Bounds should be of the form x_1,y_1,...,x_n,y_n for"
 " the bounding box [x_1,y_1] x ... x [x_n,y_n]")

parser = OptionParser(usage)
parser.add_option("--instances",type="int",dest="number_of_parallel_instances",default=1,help="Number of parallel instances of the sampling algorithm to run simultaneously.")
parser.add_option("--homparallel",type="int",dest="number_of_processors_for_bertini",default=1,help="Number of processors to use for homotopy continuation for each instance of the algorithm.")
parser.add_option("-p","--parameters",action="store_true",dest="parameter_file",default=False,help="Set this flag if you want to use parameters file. The only required argument when this flag is present is polynomial_system_directory.")
parser.add_option("--mpiexecutable",dest="mpi_executable_location",help="Use this option to specify a path to mpiexec if it is not in your PATH by defualt.")
parser.add_option("--bertini",dest="bertini_executable_location",help="Use this option to specify a path to bertini if it is not in your PATH by defualt.")
parser.add_option("--errorlimit",dest="rounding_precision",type=float,help="Specify the maximum distance of points returned by homotopy continuation from the variety.")
parser.add_option("--output",dest="output_file_name",help="Specify non-standard output file name.")
parser.add_option("--dimensionality",dest="dimensionality",type="int",help="Number of variables in polynomial system. Setting this option will check to make sure you have provided the right number of bounds.")
parser.add_option("--previous",dest="previous_points",help="Path to a previous point sample to use as smaller search space. This option must be accompanied by --prevdensity.")
parser.add_option("--prevdensity",dest="old_density",type=float,help="Density of previous sample for use with --previous.")
parser.add_option("--hosts",dest="hosts",help="Comma separated list of ssh host names to use for MPI functionality of Bertini.")
parser.add_option("--skipint",dest="skip_interval",type=float,help="Step size for increasing permissiveness of point addition heuristics. Defaults recommended.")
parser.add_option("--avglength",dest="rolling_average_length",type=int,help="Number of points to remember for point addition heuristics. Defaults recommended.")
parser.add_option("--allowedskips",dest="number_of_allowed_skips",type=int,help="Number of remembered points which are allowed to be skips before increasing permissiveness. Defaults recommended.")


(parameters,args) = parser.parse_args()
parameters = parameters.__dict__

if len(args) == 0: 
	raise RuntimeError("No polynomial system directory specified.")

template_stem = os.path.abspath(args[-1])
if not os.path.exists(template_stem): 
	raise RuntimeError("Indicated polynomial system directory does not exist.")
global_template_location = os.path.join(template_stem,"minimizer")
if not os.path.exists(global_template_location): 
	raise RuntimeError("Indicated polynomial system directory does not contain minimizer directory.")


if parameters["parameter_file"] == True: 
	parameters_path = os.path.join(template_stem,"parameters.json")
	with open(parameters_path,"r") as f: 
		parameters = json.load(f)
else: 
	if len(args) != 4: 
		raise RuntimeError("Incorrect number of arguments.")
	parameters["bounds"] = [float(entry) for entry in args[0].split(",")]
	if "dimensionality" in parameters: 
		if len(parameters["bounds"])/2 != parameters["dimensionality"]: 
			raise RuntimeError("Dimensionality and number of bounds do not match.")
	parameters["density_parameter"] = float(args[1])
	parameters["number_of_functions"] = int(args[2])



# Set up basic algorithm parameters
if "bounds" not in parameters: 
	raise RuntimeError("A bounding box must be specified.")
if "density_parameter" not in parameters: 
	raise RuntimeError("A density parameter must be specified.")
if "number_of_functions" not in parameters: 
	raise RuntimeError("The number of functions in the polynomial system must be specified.")
overall_bounds = parameters["bounds"]
overall_bounds = [float(entry) for entry in overall_bounds]
dimensionality = len(overall_bounds)/2
density_parameter = float(parameters["density_parameter"])
number_of_functions = parameters["number_of_functions"]
if "variable_indices" not in parameters: 
	variables_to_check = []
else: 
	variables_to_check = parameters["variable_indices"]
if "rounding_precision" not in parameters: 
	rounding_precision = 1e-7
else: 
	rounding_precision = parameters["rounding_precision"]
if "previous_points" not in parameters: 
	previous_run_data = False
else:
	if "old_density" not in parameters: 
		print "No previous density provided. Running without previous sample data."
		previous_run_data = False
	else: 
		previous_run_data = dict()
		previous_run_data["data"] = np.loadtxt(os.path.join(template_stem,parameters["previous_points"]),delimiter=",")
		previous_run_data["density"] = float(parameters["old_density"])

# Set up parallelization options
if "number_of_parallel_instances" not in parameters: 
	number_of_instances = 1
else: 
	number_of_instances = parameters["number_of_parallel_instances"]
computation_options = dict()
if "mpi_executable_location" in parameters: 
	computation_options["mpi_executable_location"] = parameters["mpi_executable_location"]
if "bertini_executable_location" in parameters: 
	computation_options["bertini_executable_location"] = parameters["bertini_executable_location"]
if "number_of_processors_for_bertini" in parameters: 
	computation_options["number_of_processors_for_bertini"] = parameters["number_of_processors_for_bertini"]
if "hosts" in parameters: 
	computation_options["hosts"] = parameters["hosts"]
if "skip_interval" in parameters: 
	computation_options["skip_interval"] = parameters["skip_interval"]
if "rolling_average_length" in parameters: 
	computation_options["rolling_average_length"] = parameters["rolling_average_length"]
if "number_of_allowed_skips" in parameters: 
	computation_options["number_of_allowed_skips"] = parameters["number_of_allowed_skips"]

# Splits the overall bounding box for the problem into the specified
# number of sub-boxes. 
list_of_instance_bounds = [overall_bounds]
dimension = 0
while len(list_of_instance_bounds) < number_of_instances: 
	box_to_split = list_of_instance_bounds.pop(0)
	midpoint = (float(box_to_split[2*dimension])+float(box_to_split[2*dimension+1]))/2.0
	two_halves_of_box_to_split = _splitBoxAlongDimension(box_to_split,[midpoint],dimension)
	list_of_instance_bounds += two_halves_of_box_to_split
	dimension += 1
	dimension = dimension % dimensionality

# Run all requested algorithm instances
points_queues = list()
rectangle_queues = list()
sample_rectangles_queues = list() 
process_list = list()
manager = Manager()
for box in list_of_instance_bounds: 
	points_queues += [manager.Queue()]
	rectangle_queues += [manager.Queue()]
	sample_rectangles_queues += [manager.Queue()]
	arguments = (global_template_location,number_of_functions,box,density_parameter,rounding_precision,
		points_queues[-1],rectangle_queues[-1],sample_rectangles_queues[-1],
		computation_options,False,previous_run_data,variables_to_check)
	process_list += [Process(target=sampling_algorithm,args=arguments)]
	process_list[-1].start()

print "started subprocesses"
for process in process_list: 
	process.join()

# Extract sampled points from individual algorithm instances
p_out = list()
pr_out = list() 
r_out = list()


for instance_index in range(len(list_of_instance_bounds)): 
	rectangle_queues[instance_index].put("END")
	points_queues[instance_index].put("END")
	sample_rectangles_queues[instance_index].put("END")
	for points in iter(points_queues[instance_index].get,"END"): 
		p_out += points
	for rectangles in iter(rectangle_queues[instance_index].get,"END"):
		r_out += rectangles
	for rectangles in iter(sample_rectangles_queues[instance_index].get,"END"): 
		pr_out += rectangles
p_out = np.array(p_out)

# Save the points to the specified file
# Auto formatting for floats does not work for file naming; 
# This fixes that situation; slightly hacky
density_parameter_mod = int(math.log10(density_parameter))
density_parameter_start = int(density_parameter*math.pow(10,(2-density_parameter_mod)))
if density_parameter_mod-2 < 0: 
	density_parameter_name = str(density_parameter_start)+"e"+str(density_parameter_mod-2)
else: 
	density_parameter_name = str(density_parameter_start)+"e+"+str(density_parameter_mod-2)
sample_name = density_parameter_name + "_sample.txt"
output_path = os.path.join(template_stem,sample_name)
np.savetxt(output_path,p_out,delimiter=",")







