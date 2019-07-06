from rtree import index
import numpy as np 
from math import sqrt,pow
import sys
import operator,functools,math,random
import copy

MAX_BOUNDS = float(1e6)
MAX_NUMBER_OF_OUTPUT_POINTS = 15e10

def sum_list(input_list): 
	if len(input_list) == 0: 
		return 0.0
	return np.sum(input_list)

def _doublePoint(point): 
	doubled_point = list()
	for coordinate in point: 
		doubled_point += [coordinate]
		doubled_point += [coordinate]
	return tuple(doubled_point)

def _middleOfBox(box): 
	return_point = list()
	for i in xrange(0,len(box),2):
		midpoint = (box[i]+box[i+1])/2.0
		return_point.append(midpoint)
	return return_point

def _checkRectangles(bounding_rectangle,rectangle_to_check): 
	for coordinate in xrange(0,len(bounding_rectangle),2) :
		check_low = rectangle_to_check[coordinate]
		check_high = rectangle_to_check[coordinate+1]
		bound_low = bounding_rectangle[coordinate]
		bound_high = bounding_rectangle[coordinate+1]
		if check_low < bound_low or check_high > bound_high:		
			return False
	return True

def _constructHypercube(dimension): 
	if dimension == 1: 
		return list([[0],[1]])
	zero_list = _constructHypercube(dimension-1)
	one_list = copy.deepcopy(zero_list)
	for point in zero_list: 
		point.append(0)
	for point in one_list: 
		point.append(1) 
	return zero_list + one_list 


def deinterleave(interleaved):
    dimension = len(interleaved) // 2
    def gen(): 
    	for i in xrange(dimension): 
    		yield interleaved[i]
    		yield interleaved[i+dimension]
    return list(gen())

def _matlabRectangleFormat(box): 
	box = deinterleave(box)
	if len(box)/2 == 2: 
		rstr = "patch([%1.8f,%1.8f,%1.8f,%1.8f],[%1.8f,%1.8f,%1.8f,%1.8f],\'red\')" %(box[0],box[1],box[1],box[0],box[2],box[2],box[3],box[3]) 
		rstr += "\n hold on"
	else: 
		rstr = "cubePlot([%1.8f,%1.8f,%1.8f],%1.8f,%1.8f,%1.8f,\'r\')" %(box[0],box[2],box[4],box[1]-box[0],box[3]-box[2],box[5]-box[4])
	return rstr


def _splitBox(bounds,list_of_grid_points=False):
	if not list_of_grid_points: 
		return [bounds]
	dimension = len(list_of_grid_points)
	if dimension == 1: 
		points = list_of_grid_points[0]
		return [[points[i],points[i+1]] for i in xrange(len(points)-1)]
	else: 
		output = list()
		first_dim_boxes = _splitBox(bounds,[list_of_grid_points[0]])
		second_dim_boxes = _splitBox(bounds,list_of_grid_points[1:])
		for first_box in first_dim_boxes: 
			for second_box in second_dim_boxes: 
				output.append(first_box + second_box)
		return output

def _splitBoxAlongDimension(box_to_split,input_split_points,dimension): 
	box_to_split = list(box_to_split)
	split_points = list(input_split_points)
	if dimension == 0: 
		first_half = list()
		second_half = box_to_split[2:]
	elif 2*dimension == len(box_to_split): 
		first_half = box_to_split[:-2]
		second_half = list()
	else: 
		first_half = box_to_split[:2*(dimension)]
		second_half = box_to_split[2*(dimension+1):]

	# Add in the end points if they're not already in 
	# the list
	if len(split_points) == 0: 
		split_points = [box_to_split[2*dimension],box_to_split[2*dimension+1]]
	else: 
		if split_points[0] != box_to_split[2*dimension]: 
			split_points = [box_to_split[2*dimension]] + split_points
		if split_points[-1] != box_to_split[2*dimension+1]: 
			split_points += [box_to_split[2*dimension+1]]

	new_ranges_in_given_dimension = [[split_points[i],split_points[i+1]] for i in xrange(len(split_points)-1)]
	for value_range in new_ranges_in_given_dimension: 
		if value_range[0] >= value_range[1]: 
			return False
	output_boxes = list()
	for value_range in new_ranges_in_given_dimension: 
		output_boxes += [first_half + value_range + second_half]
	return output_boxes


def _splitAlongIntersectingBox(box_to_split,box_to_intersect):
	# In this case the box to be split is contained in the box 
	# we want to split along, so just return False
	if _checkRectangles(box_to_intersect,box_to_split): 
		return False
	intersection = list() 
	for coordinate in xrange(0,len(box_to_split),2): 
		coordinates_to_add = [max(box_to_split[coordinate],box_to_intersect[coordinate]),min(box_to_split[coordinate+1],box_to_intersect[coordinate+1])]
		if coordinates_to_add[0] == coordinates_to_add[1]: 
			return box_to_split
		else: 
			intersection += [coordinates_to_add]


	output_boxes = [box_to_split]
	finished_boxes = list()
	for dimension in xrange(0,len(intersection)): 
		new_output_boxes = list()
		for box in output_boxes: 
			if _findSizeOfIntersection(box,box_to_intersect) == 0.0: 
				finished_boxes += [box]
			else:
				broken_down_box = _splitBoxAlongDimension(box,intersection[dimension],dimension)
				new_output_boxes += broken_down_box
			if broken_down_box is False: 
		output_boxes = new_output_boxes
	return finished_boxes+output_boxes

def _findIntersectionPoints(first_box,second_box): 
	intersection = list() 
	for coordinate in xrange(0,len(first_box),2): 
		coordinates_to_add = [max(first_box[coordinate],second_box[coordinate]),min(first_box[coordinate+1],second_box[coordinate+1])]
		if coordinates_to_add[0] == coordinates_to_add[1]: 
			return False
		else: 
			intersection += [coordinates_to_add]
	return intersection


def _findSizeOfIntersection(first_box,second_box): 
	intersection = list() 
	for coordinate in xrange(0,len(first_box),2): 
		if first_box[coordinate+1] < second_box[coordinate] or first_box[coordinate] > second_box[coordinate+1]: 
			return 0.0

		intersection += [max(first_box[coordinate],second_box[coordinate]),min(first_box[coordinate+1],second_box[coordinate+1])]
	return _maxLengthOfBox(intersection) 



def _findDistance(first_box,second_box): 
	point_one = np.array( _middleOfBox(first_box) )
	point_two = np.array( _middleOfBox(second_box) )
	return np.linalg.norm(point_one - point_two)

def _maxLengthOfBox(box): 
	box_lengths = np.array([(box[i+1]-box[i]) for i in xrange(0,len(box),2)])
	return np.prod(box_lengths)

def _maxSideLengthOfBox(box): 
	box_lengths = np.array([(box[i+1]-box[i]) for i in xrange(0,len(box),2)])
	# This is a dengerate case we need to check
	if np.amin(box_lengths) <= 0: 
		return 0.0
	return np.amax(box_lengths)

def _createLargeRadiusBox(point,radius): 
	range_length = radius
	box_coordinates = list()
	for coordinate in point: 
		box_coordinates.append(coordinate - range_length)
		box_coordinates.append(coordinate + range_length)
	return box_coordinates

def _checkRectangleAgainstBall(bounding_rectangle,ball): 
	center_point_of_ball =  ball.point
	total_dimension = len(bounding_rectangle)/2
	corners_of_rectangle = [[bounding_rectangle[2*dimension + hypercube_corner[dimension]] for dimension in range(total_dimension)] for hypercube_corner in _constructHypercube(total_dimension)]

	corners_of_rectangle = [np.array(corner) for corner in corners_of_rectangle]
	def distance_to_center(point): 
		return np.dot(np.array(point) - np.array(center_point_of_ball),np.array(point) - np.array(center_point_of_ball))

	distances = [distance_to_center(corner) for corner in corners_of_rectangle]

	max_distance = np.amax(distances)	

	if max_distance > ball.radius**2: 
		return False
	else:
		return True


class search_box(object): 
	def __init__(self,box,radius=0.0): 
		self.box = np.array(box)
		self.measure = _maxLengthOfBox(box)
		self.radius = radius

class indexed_box_label(object): 
	def __init__(self,point,radius,is_sample_point=False):
		self.point = point
		self.radius = radius
		self.is_sample_point = is_sample_point

class Search_Space(object): 


	def __init__(self,density_parameter,precision_parameter,global_bounds,points=None,old_distance=None,heuristics_options=None): 
		self.epsilon = density_parameter
		self.delta = precision_parameter
		p = index.Property()
		dimension = len(global_bounds)/2
		self.dimension = dimension
		p.dimension = dimension
		self.tree = index.Index(properties=p)
		self.id_counter = 0
		self.tree.interleaved = False
		self.bad_boxes = list([search_box(global_bounds)])
		self.global_bounds = global_bounds
		self.problem_bounds = list() 
		# Allows points slightly outside the given bounds to be added
		for i in range(0,len(self.global_bounds),2): 
			self.problem_bounds.append(self.global_bounds[i]-0.0*self.epsilon/2.0)
			self.problem_bounds.append(self.global_bounds[i+1]+0.0*self.epsilon/2.0)

		self.max_length_limit = _maxSideLengthOfBox(self._createSmallRadiusBox([0.0 for i in xrange(0,dimension)],self.epsilon-self.delta))
		self.current_max_length = _maxLengthOfBox(global_bounds)
		self.old_box = list([])
		self.skip_list = list()
		self.skip_radius_percentage = 1.0
		self.bad_box_reference_length = 1
		self.sample_count = 0
		self.modulous_counter = 0

		if heuristics_options is not None: 
			if heuristics_options.has_key("skip_interval"): 
				self.skip_interval = heuristics_options["skip_interval"]
			else: 
				self.skip_interval = 5.0*pow(10.0,-self.dimension)
			if heuristics_options.has_key("number_of_allowed_skips"): 
				self.number_of_allowed_skips = heuristics_options["number_of_allowed_skips"]
			else: 
				self.number_of_allowed_skips = 49 
			if heuristics_options.has_key("rolling_average_length"): 
				self.rolling_average_length = heuristics_options["rolling_average_length"]
			else: 
				self.rolling_average_length = self.number_of_allowed_skips+1
		else: 
			self.skip_interval = 5.0*pow(10.0,-self.dimension)
			self.number_of_allowed_skips = 49
			self.rolling_average_length = 50 

		if points is not None: 
			self.bad_boxes = [search_box(self._createSmallRadiusBox(point,old_distance)) for point in points]
			print "Length of bad boxes: ", len(self.bad_boxes)
			print "Old distance: ", old_distance

			for point in points: 
				self.addPoint(point,is_sample_point=True,skip_on_covered=False)

	def _createSmallRadiusBox(self,point,radius):
		range_length = radius/sqrt(self.dimension)
		box_coordinates = list()
		for coordinate in point: 
			box_coordinates.append(coordinate - range_length)
			box_coordinates.append(coordinate + range_length)
		return box_coordinates

	
	def _skipControl(self,is_skipped): 
		if len(self.skip_list) == self.rolling_average_length: 
			del self.skip_list[0]
		self.skip_list.append(int(is_skipped))

		number_of_skips = sum_list(self.skip_list)
		if number_of_skips > self.number_of_allowed_skips: 
			self.skip_radius_percentage = max(self.skip_radius_percentage - self.skip_interval,0.0)
			self.skip_list = list()
		return

	def shuffle(self): 
		random.shuffle(self.bad_boxes)
		return 

	# The default behavior (radius=False) is for inserting sample points
	# on the variety; otherwise we're inserting exclusion region boxes	
	def addPoint(self,point,radius=0,is_sample_point=False,skip_on_covered=False):
		if is_sample_point is True: 
			radius = self.epsilon
		elif radius <= self.delta:
			return
		else: 
			radius = radius - self.delta
		label = indexed_box_label(point,radius,is_sample_point)
		box = self._createSmallRadiusBox(point,radius)
		if is_sample_point is True:
			# Prevents adding sample points outside of the bounds of the 
			# problem
			if _checkRectangles(self.problem_bounds,_doublePoint(point)) == False: 
				return
			# Prevents adding wholly unnecessary sample points
			if skip_on_covered is True:
				intersecting_objects = list(self.tree.intersection(box,objects=True))
				intersecting_boxes = [item for item in intersecting_objects if item.object.is_sample_point is True]
				if len(intersecting_boxes) != 0: 
					intersecting_boxes = [deinterleave(item.bbox) for item in intersecting_boxes]
					box_sizes = [_findDistance(box,bbox) for bbox in intersecting_boxes]
					max_box_size = np.amin(box_sizes)
					if max_box_size <= 2.0*self.skip_radius_percentage*self.epsilon and self.skip_radius_percentage > 0.0: 
						self._skipControl(True)
						return 
					if self.checkCover([box]) == True: 
						return
				self._skipControl(False)
				self.sample_count += 1
		self.tree.insert(self.id_counter,_createLargeRadiusBox(point,radius),label)
		self.id_counter += 1

		modulus = 300
		if (self.modulous_counter % modulus == 0 and len(self.bad_boxes) > 0): 
			box = self.bad_boxes.pop(np.argmax([box.measure for box in self.bad_boxes]))
			self.bad_boxes = self.bad_boxes + [box]
			# self.current_max_length = self.bad_boxes[-1].measure
			# self.bad_box_reference_length = len(self.bad_boxes)
		return 


	def _checkIfBoxIsCovered(self,input_bounds):
		bounds = list(input_bounds.box)
		intersecting_balls = list(self.tree.intersection(bounds,objects=True))
	
		if len(intersecting_balls)==0: 
			return False
		
		intersecting_balls = [item.object for item in intersecting_balls]		
		intersecting_boxes = [self._createSmallRadiusBox(ball.point,ball.radius) for ball in intersecting_balls]

		for ball in intersecting_balls: 
			if _checkRectangleAgainstBall(bounds,ball) is True:
				return True
		# In some edge cases, numerical instability in calculating ball containment
		# leads to incorrect results. The following check is less permissive
		# in general but more stable in those instances.
		for box in intersecting_boxes: 
			if _checkRectangles(box,bounds): 
				return True

		intersection_sizes = [_findSizeOfIntersection(bounds,box) for box in intersecting_boxes]
		index = np.argmax(intersection_sizes)
		if intersection_sizes[index] <= 0.0: 
			return False 
		box_to_split_along = intersecting_boxes[index]
		split_up_box = _splitAlongIntersectingBox(bounds,box_to_split_along)

		if split_up_box is False: 
			return False 

		return [search_box(box) for box in split_up_box]

	# The idea: Split the search space into number_of_processors rectangular regions 
	# In parallel, run this recursive search: 
	# (1) Does the current rectangle intersect any rectangles in the tree? If not, return False
	# (2) If so, is the current rectangle strictly contained in any of the tree rectangles? If so, 
	# return true. If not, break the current rectangle into smaller rectangles and check the 
	# smaller rectangles
	# Note: This search is breadth first- we want to stop searching at the largest 
	# "bad" rectangle, since we otherwise could waste a lot of time confirming a good recatangle
	def checkCover(self,bad_boxes=None):
		bbflag = False
		index_max = 0
		if bad_boxes is None: 
			if self.sample_count >= MAX_NUMBER_OF_OUTPUT_POINTS: 
				return True
			bad_boxes = self.bad_boxes 
			bbflag = True
		else: 
			bad_boxes = [search_box(box) for box in bad_boxes]


		while len(bad_boxes)>0:
			box = bad_boxes.pop() 
			intersection_status = self._checkIfBoxIsCovered(box) 
			self.modulous_counter += 1
			if hasattr(intersection_status,"__iter__"): 
				bad_boxes = intersection_status+bad_boxes
				if bbflag is True: 
					self.bad_boxes = bad_boxes
			elif intersection_status is False:
				bad_boxes = [box]+bad_boxes
				if bbflag is True: 
					self.bad_boxes = bad_boxes
				return _middleOfBox(box.box)

		# If we reached this point then there are no bad boxes, so 
		# the range is covered
		return True 

	def outputSamplePoints(self): 
		absolute_bounds = list()
		for i in xrange(0,self.dimension): 
			absolute_bounds.append(-MAX_BOUNDS)
			absolute_bounds.append(MAX_BOUNDS)
		all_values = self.tree.intersection(absolute_bounds,objects=True)
		output_values = [] 
		for item in all_values: 
			if item.object.is_sample_point is True: 
				output_values += [item.object.point]
		return output_values

	def outputSampleRectangles(self): 
		absolute_bounds = list()
		for i in xrange(0,self.dimension): 
			absolute_bounds.append(-MAX_BOUNDS)
			absolute_bounds.append(MAX_BOUNDS)
		all_values = self.tree.intersection(absolute_bounds,objects=True)
		all_values = [_matlabRectangleFormat(item.bbox) for item in all_values if item.object.is_sample_point==True]
		return all_values

	def outputMinPoints(self): 
		absolute_bounds = list()
		for i in xrange(0,self.dimension): 
			absolute_bounds.append(-MAX_BOUNDS)
			absolute_bounds.append(MAX_BOUNDS)
		all_values = self.tree.intersection(absolute_bounds,objects=True)
		all_values = [item.object.point for item in all_values if item.object.is_sample_point!=True]
		return all_values

	def outputMinPointsMatlab(self): 
		absolute_bounds = list()
		for i in xrange(0,self.dimension): 
			absolute_bounds.append(-MAX_BOUNDS)
			absolute_bounds.append(MAX_BOUNDS)
		all_values = self.tree.intersection(absolute_bounds,objects=True)
		all_values = [_matlabRectangleFormat(item.bbox) for item in all_values if item.object.is_sample_point!=True]
		return all_values