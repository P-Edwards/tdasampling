import numpy as np 
import tempfile
import os
import sys
from os.path import join
from shutil import copyfile,rmtree
from subprocess import call
from multiprocessing import Process, Queue



norm = np.linalg.norm
IMAGINARY_PART_THRESHOLD = 1e-7
MAX_TIME = 20*60 # Maximum number of seconds that a homotopy can run before it times out


def filterImaginaryPart(complex_pair): 
	for coordinate in complex_pair: 
		if abs(coordinate[1]) > IMAGINARY_PART_THRESHOLD: 
			return False 
	return True


def readSolutionsFile(solutions_path,dimensionality,indices):
	solution_points = []
	
	# This allows us to pick out the variables at specified indices, 
	# rather than restricting to the first <dimensionality> of them
	if len(indices) == 0: 
		indices = range(dimensionality)

	if not os.path.isfile(solutions_path): 
		return []
	# Read in solution points
	solutions_file = open(solutions_path,"r") 
	lines = solutions_file.readlines() 
	solutions_file.close()

	number_of_solutions = int(lines[0])


	# This shouldn't happen, but sometimes thresholds don't work
	if number_of_solutions == 0:
		return solution_points;

	# Removes unnecessary lines at the beginning of the file
	lines = lines[2:]
	coordinate = 0 
	current_point = []
	line_number = 0

	while line_number < len(lines): 
		# This triggers if we've read in all the non-auxiliary variables
		if len(solution_points) == number_of_solutions: 
			break 
		if len(current_point) == dimensionality:
			coordinate = 0 
			solution_points += [current_point] 
			current_point = []
			# Skip past lines with auxiliary variables
			while (lines[line_number]!='\n'):
				line_number += 1
			# And then move to the beginning of the next solution point block
			while (line_number < len(lines)-1 and lines[line_number] == '\n'): 
				line_number += 1 
			continue
		if coordinate in indices: 
			solution_fragment = lines[line_number]
			solution_fragment = solution_fragment.split(" ")
			real_part = float(solution_fragment[0])
			imaginary_part = float(solution_fragment[1])
			current_point  += [(real_part,imaginary_part)]
		coordinate += 1
		line_number += 1
	array_to_return = [ [coordinate[0] for coordinate in point] for point in solution_points if filterImaginaryPart(point)] 
	array_to_return = np.array(array_to_return)
	return array_to_return

def bertiniMinimizer(run_path,input_point,number_of_bertini_processes,number_of_functions,dimensionality,bertini_executable,mpi_executable,space_bounds,hosts=None,variable_indices=[]):
	# final_parameters contains only information about the test point
	# One parameter per variable, plus one if we're doing the smoothing thing
	def constructFinalParametersString(final_parameters,parameters_are_complex,number_of_functions):
		output_string = str((dimensionality+number_of_functions)) + "\n\n"
		for parameter in final_parameters: 
			if parameters_are_complex:
				output_string += str(parameter[0]) + " " + str(parameter[1]) + "\n"
			else: 
				output_string += str(parameter) + " " + "0\n"
		for dummy_variable in xrange(number_of_functions):
			output_string += "0 0\n"
		return output_string

	final_parameters_path = join(run_path,"final_parameters")

	solutions_file_path = join(run_path,"finite_solutions")
	devnull = open(os.devnull,"w")

	final_parameters_file = open(final_parameters_path,"w")
	final_parameters_file.write(constructFinalParametersString(input_point,False,number_of_functions))
	final_parameters_file.close()
	if hosts is None: 
		call(args=["timeout",str(MAX_TIME)+"s",mpi_executable,"-np", str(number_of_bertini_processes), bertini_executable,"input_param","start_points"],stdout=devnull.fileno(),cwd=run_path)
	else: 
		call(args=["timeout",str(MAX_TIME)+"s",mpi_executable,"-np", str(number_of_bertini_processes),"-hosts",hosts,bertini_executable,"input_param","start_points"],stdout=devnull.fileno(),cwd=run_path)

	devnull.close()

	solution_points = readSolutionsFile(solutions_file_path,dimensionality,variable_indices)
	if len(solution_points) == 0: 
		print "Failed to find solution points for input point: ",input_point
		return {"points": False, "distance": 0.0}
	solution_distances = np.array([norm(input_point-critical_point) for critical_point in solution_points])
	min_index = np.argmin(solution_distances)

	return {"points": solution_points, "distance": solution_distances[min_index],"min_point": solution_points[min_index] }
 
def bertiniMinimizerHRC(run_path,input_point,number_of_bertini_processes,number_of_functions,dimensionality,bertini_executable,mpi_executable,space_bounds,hosts=None,variable_indices=[]):
	# final_parameters contains only information about the test point
	# One parameter per variable, plus one if we're doing the smoothing thing
	def constructFinalParametersString(final_parameters,parameters_are_complex,number_of_functions):
		output_string = str((dimensionality+number_of_functions)) + "\n\n"
		for parameter in final_parameters: 
			if parameters_are_complex:
				output_string += str(parameter[0]) + " " + str(parameter[1]) + "\n"
			else: 
				output_string += str(parameter) + " " + "0\n"
		for dummy_variable in xrange(number_of_functions):
			output_string += "0 0\n"
		return output_string

	final_parameters_path = join(run_path,"final_parameters")

	solutions_file_path = join(run_path,"finite_solutions")
	devnull = open(os.devnull,"w")

	final_parameters_file = open(final_parameters_path,"w")
	final_parameters_file.write(constructFinalParametersString(input_point,False,number_of_functions))
	final_parameters_file.close()

	call(args=["timeout",str(MAX_TIME)+"s","srun --mpi=pmi2",bertini_executable,"input_param","start_points"],stdout=devnull.fileno(),cwd=run_path)		

	devnull.close()

	solution_points = readSolutionsFile(solutions_file_path,dimensionality,variable_indices)
	if len(solution_points) == 0: 
		print "Failed to find solution points for input point: ",input_point
		return {"points": False, "distance": 0.0}
	solution_distances = np.array([norm(input_point-critical_point) for critical_point in solution_points])
	min_index = np.argmin(solution_distances)

	return {"points": solution_points, "distance": solution_distances[min_index],"min_point": solution_points[min_index] }



def bertiniEval(run_path,points,number_of_bertini_processes,number_of_functions,dimensionality,bertini_executable,mpi_executable,variable_indices=[]):
	start_path = join(run_path,"start")
	solutions_file_path = join(run_path,"function")

	output_string = str((len(points))) + "\n\n"
	for point in points:
		for coordinate in point:  
			output_string += str(coordinate) + " " + "0\n"

	devnull = open(os.devnull,"w")
	start_file = open(start_path,"w")
	start_file.write(output_string)
	start_file.close()	
	call(args=[mpi_executable,"-np", str(number_of_bertini_processes), bertini_executable],stdout=devnull.fileno(),cwd=run_path)
	devnull.close()
	# The second argument here is number of functions instead of the dimensionality
	# since evaluating number_of_functions polynomials on a single point produces that 
	# many answers
	evaluated_points = readSolutionsFile(solutions_file_path,number_of_functions,variable_indices)
	return evaluated_points





