import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from scipy import misc
from Queue import Queue
from collections import deque

img = misc.imread('/Users/Nikhil/Desktop/Canny_pictures/lena.png')

#Constants
negative_infinity = float("-inf")
infinity = float("inf")
smallSlope = .414
bigSlope = 2.414
y_length = len(img)
x_length = len(img[0])
black = 0
white = 255
high_threshold = 175
low_threshold = 75
gaussianSmoothMask_Kernel = np.array([[2, 4,  5,  4,  2],
									  [4, 9,  12, 9,  4],
									  [5, 12, 15, 12, 5],
									  [4, 9,  12, 9,  4],
									  [2, 4,  5,  4,  2]])

vertical_sobel_mask = np.array([[-1, 0, 1],
								[-2, 0, 2],
								[-1, 0, 1]])

horizontal_sobel_mask = np.array([[-1, -2, -1],
								  [0,   0,  0],
								  [1,   2,  1]])

visited = set()

#Directions
S  = (0, -1)
SE = (1, -1)
E  = (1, 0)
NE = (1, 1)
N  = (0, 1)
NW = (-1, 1)
W  = (1, 0)
SW = (1, -1)

Directions = set([S, SE, E, NE, N, NW, W, SW])


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def display(array):
	plt.imshow(array, cmap = plt.get_cmap('gray'))
	plt.show()

def gaussianSmoothMask(array):
	Kernel = np.divide(gaussianSmoothMask_Kernel, 115.)
	return signal.convolve2d(array, Kernel, boundary='symm', mode='same')

def verticalEdges(array):
	return signal.convolve2d(array, vertical_sobel_mask,boundary='symm', mode='same')

def horizontalEdges(array):
	return signal.convolve2d(array, horizontal_sobel_mask,boundary='symm', mode='same')

# It is important to remember here, that the horizontal sobel mask finds vertical gradients
# and the vertical sobel mask finds horizontal gradients, so when inputting do not forget
# to switch them
def computeDirection(verticalGradient, horizontalGradient):
	
	if (horizontalGradient == 0):
		slope = infinity * np.sign(verticalGradient)
	else:
		slope = verticalGradient/horizontalGradient


	if (slope <= smallSlope and slope >= -smallSlope):
		return (1,0)
	elif (slope > smallSlope and slope < bigSlope):
		return (1,1)
	elif (slope < -smallSlope and slope > -bigSlope):
		return (1, -1)
	elif (slope >= bigSlope):
		return (0,1)
	else:
		return (0, -1)

def combineGradients(vertical_edges, horizontal_edges):
	y_counter = 0
	gradient_array = np.zeros((y_length, x_length))
	direction_array = np.zeros((y_length, x_length, 2))
	while(y_counter < y_length):
		x_counter = 0
		while(x_counter < x_length):
			v_value = vertical_edges[y_counter][x_counter]
			h_value = horizontal_edges[y_counter][x_counter]
			gradient_array[y_counter][x_counter] = np.sqrt(v_value**2 + h_value**2)
			direction_array[y_counter][x_counter] = computeDirection(h_value, v_value)
			x_counter += 1
		y_counter += 1
	return gradient_array , direction_array

def computeGradients(array):
	vertical_edges = verticalEdges(array)
	horizontal_edges = horizontalEdges(array)
	return combineGradients(vertical_edges, horizontal_edges)

def localMaximum(x, y, gradients, step):
	x_step = step[0]
	y_step = step[1]
	localValue = gradients[y][x]
	if (outOfBounds(x + x_step, y + y_step) or outOfBounds(x - x_step, y - y_step)):
		return False
	elif (localValue > gradients[y + y_step][x + x_step] and localValue > gradients[y - y_step][x -x_step]):
		return True
	return False

def outOfBounds(x, y):
	if (x < 0 or x >= x_length):
		return True
	elif (y < 0 or y >= y_length):
		return True
	return False


def nonMaximumSuppress(gradients, directions):
	suppressed_array = np.zeros((y_length, x_length))
	y_counter = 0
	while(y_counter < y_length):
		x_counter = 0
		while(x_counter < x_length):
			step = directions[y_counter][x_counter]
			if (localMaximum(x_counter, y_counter, gradients, step)):
				suppressed_array[y_counter][x_counter] = gradients[y_counter][x_counter]
			x_counter += 1
		y_counter += 1
	return suppressed_array

def backgroundPixelRemove(value):
	if not value == white:
		return black
	return white

finishImage = np.vectorize(backgroundPixelRemove)

def vibratingPixel(value):
	if (value > low_threshold and value <= high_threshold):
		return True
	return False

def highValue(value):
	if (value > high_threshold):
		return True
	return False

# def surroundedHigh(array, x, y):
# 	surrounded = False
# 	for dy in steps:
# 		for dx in steps:
# 			value = array[y + dy][x + dx]
# 			if vibratingPixel(value):
# 				arr

# def visit(array, x, y):
# 	value = array[y][x]
# 	if backgroundPixel(value):
# 		array[y][x] = black
# 	elif vibratingPixel(array, x, y):


# def hytherisisThreshold(array):
# 	y = 0
# 	while (y < y_length):
# 		x = 0
# 		while (x < x_length):
# 			coordinate = (x, y)
# 			if not coordinate in visited:
# 				#visit(array, x_counter, y_counter)
# 				value = array[y][x]
# 				if backgroundPixel(value):
# 					array[y][x] = black
# 				# else:
# 				# 	array[y][x] = white
# 				elif highValue(value):
# 					array[y][x] = white
# 					visited.add((x,y))
# 					for dx,dy in Directions
# 						x_coor = x + dx
# 						y_coor = y + dy
# 						if not outOfBounds(x_coor, y_coor):
# 							if vibratingPixel(array[y_coor][x_coor]):
# 								array[y_coor][x_coor] = white
# 								visited.add((x_coor,y_coor))
# 			x += 1
# 		y += 1
# 	return array

def getChildren(x, y):
	children = []
	for dx, dy in Directions:
		newy = y + dy
		newx = x + dx
		newCoor = (newx,newy)
		if not outOfBounds(newx, newy) and not newCoor in visited:
			children += [newCoor]
			visited.add(newCoor)
	return children

def visit(x, y, array):
	if not outOfBounds(x, y):
		visited.add((x,y))
		array[y][x] = white
		fringe = deque([])
		for child in getChildren(x,y):
			fringe.appendleft(child)
		while not len(fringe) == 0:
			next = fringe.pop()
			nx = next[0]
			ny = next[1]
			value = array[ny][nx]
			if highValue(value):
				array[ny][nx] = white
				for child in getChildren(nx, ny):
					fringe.appendleft(child)
			elif vibratingPixel(value):
				array[ny][nx] = white
				for child in getChildren(nx, ny):
					fringe.appendleft(child)
	return array


def connectEdge(array):
	y = 0
	while (y < y_length):
		x = 0
		while (x < x_length):
			coordinate = (x,y)
			if not coordinate in visited:
				value = array[y][x]
				if highValue(value):
					array = visit(x, y, array)
			x += 1
		y += 1
	return array

def genThreshold(array):
	y = 0
	while (y < y_length):
		x = 0
		while (x < x_length):
			value = array[y][x]
			if highValue(value):
				array[y][x] = white
			else:
				array[y][x] = black
			x += 1
		y += 1
	return array

def hysteresisThreshold(array):
	edgeConnected = connectEdge(array)
	final = finishImage(edgeConnected)
	return final



gray = rgb2gray(img)    

fuzzy = gaussianSmoothMask(img)

gradients, directions = computeGradients(fuzzy)

suppressed = nonMaximumSuppress(gradients, directions)

final = hysteresisThreshold(suppressed)

display(final)






