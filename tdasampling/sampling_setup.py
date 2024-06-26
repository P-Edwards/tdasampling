from sympy import symbols,diff
from sympy.parsing.sympy_parser import parse_expr
from subprocess import call
import os


def sampling_setup(parameters, args):
	setup_directory_path = os.path.abspath(args[0])

	system_path = os.path.join(setup_directory_path,"polynomial_system")
	system_file = open(system_path,"r") 
	lines = system_file.readlines() 
	system_file.close()

	variable_string = lines[0]
	polynomial_system = lines[1:]



	polynomial_system = [polynomial.replace("^","**") for polynomial in polynomial_system]
	polynomial_system = [parse_expr(polynomial) for polynomial in polynomial_system]


	variables = [symbols(variable) for variable in variable_string.split(",")]
	variable_parameters = [symbols(variable+"par") for variable in variable_string[:-1].split(",")]

	homogeneous_variables = [symbols("l"+str(i)) for i in range(len(polynomial_system)+1)]
	function_parameters = [symbols("eps"+str(i)) for i in range(len(polynomial_system))]

	gradients = list()
	for polynomial in polynomial_system: 
		grad = [diff(polynomial,variable) for variable in variables]
		gradients.append(list(grad))

	final_system_polynomials = [polynomial_system[i] - function_parameters[i] for i in range(len(polynomial_system))]
	final_system_gradient_polynomials = list()
	for variable_index in range(len(variables)): 
		base = homogeneous_variables[0]*(variables[variable_index] - variable_parameters[variable_index])
		for polynomial_index in range(len(polynomial_system)): 
			base += homogeneous_variables[polynomial_index+1]*gradients[polynomial_index][variable_index]
		final_system_gradient_polynomials.append(base)

	final_system_polynomials = [" f"+str(i) + " = " + str(final_system_polynomials[i]).replace("**","^") +";" for i in range(len(final_system_polynomials))]
	final_system_gradient_polynomials = [" g"+ str(i) + " = " + str(final_system_gradient_polynomials[i]).replace("**","^") + ";" for i in range(len(final_system_gradient_polynomials))]


	minimizer_path = os.path.join(setup_directory_path,"minimizer")
	if not os.path.exists(minimizer_path): 
		os.makedirs(minimizer_path)
	input_start_path = os.path.join(minimizer_path,"input_start")

	with open(input_start_path,"w") as input_start_file: 
		input_start_file.write("CONFIG\n"
		+ "ParameterHomotopy: 1\n"
		+ "END; \n"
		+ "INPUT\n"
		+ "parameter " + ",".join([str(parameter) for parameter in variable_parameters] + [str(param) for param in function_parameters]) + ";\n"
		+ "variable_group " + ",".join([str(variable) for variable in variables]) + ";\n"
		+ " hom_variable_group " + ",".join([str(hom_variable) for hom_variable in homogeneous_variables]) + ";\n"
		+ " function " + ",".join(["f"+str(i) for i in range(len(final_system_polynomials))] + ["g"+str(i) for i in range(len(final_system_gradient_polynomials))]) + ";\n"
		+ "\n".join(final_system_polynomials) 
		+ "\n" 
		+ "\n".join(final_system_gradient_polynomials)
		+ "\n"
		+ "END;"
		)

	if parameters["mpi_executable_location"] != "mpiexec":
		mpi_executable_location = os.path.abspath(parameters["mpi_executable_location"])
	else: 
		mpi_executable_location = "mpiexec"
	if parameters["bertini_executable_location"] != "bertini":
		bertini_executable_location = os.path.abspath(parameters["bertini_executable_location"])
	else:
		bertini_executable_location = "bertini"
	number_of_bertini_processes = parameters["number_of_bertini_processes"]

	with open(os.devnull,"w") as devnull:
		if parameters["hosts"] is not None: 
			call(args=[mpi_executable_location,"-np", str(number_of_bertini_processes),"-hosts",parameters["hosts"],bertini_executable_location,"input_start"],stdout=devnull.fileno(),cwd=minimizer_path)
		else: 
			if number_of_bertini_processes == 1: 
				call(args=[bertini_executable_location,"input_start"],stdout=devnull.fileno(),cwd=minimizer_path)
			else: 	
				call(args=[mpi_executable_location,"-np", str(number_of_bertini_processes),bertini_executable_location,"input_start"],stdout=devnull.fileno(),cwd=minimizer_path)

	os.rename(os.path.join(minimizer_path,"nonsingular_solutions"),os.path.join(minimizer_path,"start_points"))

	input_param_path = os.path.join(minimizer_path,"input_param")
	with open(input_param_path,"w") as input_param_file: 
		input_param_file.write("CONFIG\n"
		+ "ParameterHomotopy: 2\n"
		+ "END; \n"
		+ "INPUT\n"
		+ "parameter " + ",".join([str(parameter) for parameter in variable_parameters] + [str(param) for param in function_parameters]) + ";\n"
		+ "variable_group " + ",".join([str(variable) for variable in variables]) + ";\n"
		+ " hom_variable_group " + ",".join([str(hom_variable) for hom_variable in homogeneous_variables]) + ";\n"
		+ " function " + ",".join(["f"+str(i) for i in range(len(final_system_polynomials))] + ["g"+str(i) for i in range(len(final_system_gradient_polynomials))]) + ";\n"
		+ "\n".join(final_system_polynomials) 
		+ "\n" 
		+ "\n".join(final_system_gradient_polynomials)
		+ "\n"
		+ "END;"
		)

