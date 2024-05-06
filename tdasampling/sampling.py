from tdasampling import sampling_algorithm
from tdasampling.search_space import _splitBoxAlongDimension
from multiprocessing import Process,Manager
import multiprocessing
import math,sys,os
import numpy as np
import json


def wrap_sampling_algorithm(abc): 
	return sampling_algorithm(*abc)



def sampling(parameters, args):
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
			parameters2 = json.load(f)
		for key in list(parameters.keys()): 
			if key not in parameters2: 
				parameters2[key] = parameters[key]
		parameters = parameters2
	else: 
		if len(args) != 4: 
			raise RuntimeError("Incorrect number of arguments.")
		parameters["bounds"] = [float(entry) for entry in args[0].split(",")]
		if parameters["dimensionality"] is not None: 
			if len(parameters["bounds"])/2 != parameters["dimensionality"]: 
				raise RuntimeError("Dimensionality and number of bounds do not match.")
		parameters["density_parameter"] = float(args[1])
		parameters["number_of_functions"] = int(args[2])
	
	
	
	# Set up basic algorithm parameters
	if parameters["bounds"] is None: 
		raise RuntimeError("A bounding box must be specified.")
	if parameters["density_parameter"] is None: 
		raise RuntimeError("A density parameter must be specified.")
	if parameters["number_of_functions"] is None:
		raise RuntimeError("The number of functions in the polynomial system must be specified.")
	overall_bounds = parameters["bounds"]
	overall_bounds = [float(entry) for entry in overall_bounds]
	dimensionality = len(overall_bounds)//2
	density_parameter = float(parameters["density_parameter"])
	number_of_functions = parameters["number_of_functions"]
	if parameters["variable_indices"] is not None:
		variables_to_check = [int(variable) for variable in str(parameters["variable_indices"]).split(",")]
	else: 
		variables_to_check = []
	rounding_precision = parameters["rounding_precision"]
	if parameters["previous_points"] is None: 
		previous_run_data = False
	else:
		if parameters["old_density"] is not None: 
			print("No previous density provided. Running without previous sample data.")
			previous_run_data = False
		else: 
			previous_run_data = dict()
			previous_run_data["data"] = np.loadtxt(os.path.join(template_stem,parameters["previous_points"]),delimiter=",")
			previous_run_data["density"] = float(parameters["old_density"])
	
	# Set up parallelization options
	number_of_instances = parameters["number_of_parallel_instances"]
	computation_options = dict()
	if parameters["mpi_executable_location"] is not None: 
		computation_options["mpi_executable_location"] = parameters["mpi_executable_location"]
	if parameters["bertini_executable_location"] is not None: 
		computation_options["bertini_executable_location"] = parameters["bertini_executable_location"]
	if parameters["number_of_processors_for_bertini"] is not None: 
		computation_options["number_of_processors_for_bertini"] = parameters["number_of_processors_for_bertini"]
	if parameters["hosts"] is not None: 
		computation_options["hosts"] = parameters["hosts"]
	if parameters["skip_interval"] is not None: 
		computation_options["skip_interval"] = parameters["skip_interval"]
	if parameters["rolling_average_length"] is not None: 
		computation_options["rolling_average_length"] = parameters["rolling_average_length"]
	if parameters["number_of_allowed_skips"] is not None:
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
	process_argument_list = list()
	manager = Manager()
	for box in list_of_instance_bounds: 
		points_queues += [manager.Queue()]
		rectangle_queues += [manager.Queue()]
		sample_rectangles_queues += [manager.Queue()]
		arguments = [global_template_location,number_of_functions,box,density_parameter,rounding_precision,
			points_queues[-1],rectangle_queues[-1],sample_rectangles_queues[-1],
			computation_options,False,previous_run_data,variables_to_check]
		process_argument_list += [arguments]
	
		
	if parameters["total_number_of_processors"] is None: 
		total_processors = multiprocessing.cpu_count()//2
	else: 
		total_processors = parameters["total_number_of_processors"]
	p = multiprocessing.Pool(total_processors)
	# The map_async and get timeout is a bug workaround for getting ctrl+C to work correctly. 
	# See https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
	process_pool = p.map_async(wrap_sampling_algorithm,process_argument_list).get(9e8)
	# process_pool.wait()
	
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
	if parameters["output_path"] is not None: 
		output_path = os.path.abspath(parameters["output_path"])
	else: 
		output_path = os.path.join(template_stem,sample_name)
	np.savetxt(output_path,p_out,delimiter=",")
	