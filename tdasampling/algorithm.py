import numpy as np
from search_space import Search_Space,_splitBoxAlongDimension
from walking import bertiniMinimizer,bertiniEval
import multiprocessing
import os,sys
import resource
from shutil import rmtree,copyfile
import tempfile
from os.path import join
from math import sqrt
import math

Process = multiprocessing.Process
ss = Search_Space

MAXIMUM_NUMBER_OF_POINTS_TO_ADD = 20


def check_for_nonzero_entry(input_list): 
	for entry in input_list:
		if entry > 0: 
			return True
	return False




def sampling_algorithm(global_template_location,number_of_functions,bounds,density_parameter,rounding_precision,points_queue,rectangles_queue,sample_rectangles_queue,computation_options=dict(),eval_template_location=False,smaller_box_info=False,variable_indices=[]):
	# These files are required for the algorithm
	if not os.path.exists(os.path.join(global_template_location,"input_param")): 
		raise RuntimeError("No parameter homotopy file input_param found in directory " + global_template_location)
	if not os.path.exists(os.path.join(global_template_location,"start_points")): 
		raise RuntimeError("No starting solutions found for parameter homotopy in file start_points.")
	if not os.path.exists(os.path.join(global_template_location,"start_parameters")): 
		raise RuntimeError("No starting parameters found for parameter homotopy in minimizer directory.")
	
	if computation_options.has_key("mpi_executable_location"): 
		mpi_executable_location = computation_options["mpi_executable_location"]
	else: 
		mpi_executable_location = "mpirun"

	if computation_options.has_key("bertini_executable_location"): 
		bertini_executable_location = computation_options["bertini_executable_location"]
	else: 
		bertini_executable_location = "bertini"

	if computation_options.has_key("number_of_processors_for_bertini"): 
		number_of_processors_for_bertini = computation_options["number_of_processors_for_bertini"]
	else: 
		number_of_processors_for_bertini = multiprocessing.cpu_count()/2

	if computation_options.has_key("hosts"): 
		distributed_run_flag = True
		hosts = computation_options["hosts"]
	else: 
		distributed_run_flag = False

	if computation_options.has_key("skip_interval") or computation_options.has_key("rolling_average_length") or computation_options.has_key("number_of_allowed_skips"): 
		heuristics_options = dict()
		if computation_options.has_key("skip_interval"): 
			heuristics_options["skip_interval"] = computation_options["skip_interval"]
		if computation_options.has_key("rolling_average_length"): 
			heuristics_options["rolling_average_length"] = computation_options["rolling_average_length"]
		if computation_options.has_key("number_of_allowed_skips"): 
			heuristics_options["number_of_allowed_skips"] = computation_options["number_of_allowed_skips"]
	else: 
		heuristics_options = None

	if smaller_box_info: 	
		space = ss(density_parameter,rounding_precision,bounds,smaller_box_info["data"],smaller_box_info["density"],heuristics_options)
	else: 
		space = ss(density_parameter,rounding_precision,bounds,heuristics_options=heuristics_options)
	dimensionality = space.dimension
	

	if distributed_run_flag: 
		run_path = os.path.join(global_template_location,"computation_directory")
		directory_counter = 0 
		while os.path.exists(run_path+str(directory_counter)): 
			directory_counter += 1 
		run_path = run_path + str(directory_counter)
		def minimizer(point_to_test): 
			return bertiniMinimizer(run_path,point_to_test,number_of_processors_for_bertini,number_of_functions,dimensionality,bertini_executable_location,mpi_executable_location,bounds,hosts,variable_indices)
	else: 
		run_path = tempfile.mkdtemp()
		def minimizer(point_to_test): 
			return bertiniMinimizer(run_path,point_to_test,number_of_processors_for_bertini,number_of_functions,dimensionality,bertini_executable_location,mpi_executable_location,bounds,variable_indices=variable_indices)

	if not os.path.exists(run_path):
		os.makedirs(run_path)

	if eval_template_location != False: 
		eval_path = join(run_path,"eval")
		os.makedirs(eval_path)
		eval_configuration = join(eval_template_location,"input")
		copyfile(eval_configuration,join(eval_path,"input"))


	configuration_file = join(global_template_location,"input_param")
	input_solutions = join(global_template_location,"start_points")
	start_parameters = join(global_template_location,"start_parameters")



	copyfile(configuration_file,join(run_path,"input_param"))
	copyfile(input_solutions,join(run_path,"start_points"))
	copyfile(start_parameters,join(run_path,"start_parameters"))


	test_point = space.checkCover()
	old_test_point = test_point

	# Only start adding sample points once the exclusion boxes have gotten uselessly small
	# in order to reduce number of samples

	

	# This counter controls the progress output
	counter = 1
	while test_point!=True:
		min_distance_results =  minimizer(test_point)
		
		# It's highly unlikely but technically possible to hit an exceptional test point 
		# value. This handles that case.
		if min_distance_results['points'] is False: 
			# Pick random unit vector    
			rand_vector = np.random.normal(size=dimensionality)
			rand_vector = rand_vector/np.linalg.norm(rand_vector)
			test_point = np.array(test_point) + rand_vector
			test_point = list(test_point)
			continue

		space.addPoint(test_point,min_distance_results['distance'])
		
		# Also add critical points to the sampling up to 
		# a fixed maximum
		points = min_distance_results['points']
		if len(points) > MAXIMUM_NUMBER_OF_POINTS_TO_ADD: 
			points = points[:MAXIMUM_NUMBER_OF_POINTS_TO_ADD]

		# Filtering for semialgebraic runs
		if eval_template_location != False: 
			evaluated_points = bertiniEval(eval_path,points,number_of_min_processes,number_of_eval_functions,dimensionality,bertini_executable_location,mpi_executable_location)
			filtered_points = [points[index] for index in xrange(len(points)) if check_for_nonzero_entry(evaluated_points[index])]
			points = filtered_points

		for point in points: 
			space.addPoint(point,is_sample_point=True,skip_on_covered=True)
		# Breaks out of loops where we're getting the same test point over and over
		old_test_point = test_point
		test_point = space.checkCover()
		while old_test_point == test_point and len(space.bad_boxes) > 1: 
			space.shuffle()
			old_test_point = test_point
			test_point = space.checkCover()

		counter += 1
		if counter % 100 == 0: 
			print "\n The algorithm instance checking bounds: ", space.global_bounds,"has boxes remaining: ", len(space.bad_boxes), "\n"



	rmtree(run_path)

	sample_points = space.outputSamplePoints()
	if len(sample_points) > 0: 
		points_queue.put([list(point) for point in sample_points])

	sample_rectangles = space.outputSampleRectangles()
	sample_rectangles_queue.put(sample_rectangles)
	min_rectangles = space.outputMinPointsMatlab()
	rectangles_queue.put(min_rectangles)


