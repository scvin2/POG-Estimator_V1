from constants import *


# Fucntion for scaling down the value between 0 to 1 range or -1 to 1 range
def scale_down_value(value, min_value, max_value):  # choose the scale down format here
	if SCALING == "minmax":
		return scale_down_minmax(value, min_value, max_value)
	elif SCALING == "m1p1":
		return scale_down_one_to_one(value, min_value, max_value)


# scales down given data to 0 to 1 range
def scale_down_minmax(value, min_value, max_value):  # min max scaling [0,1]
	return (value-min_value)/(max_value-min_value)


# scales down given data to -1 to +1 range
def scale_down_one_to_one(value, min_value, max_value):  # scaling to [-1,1]
	return (2 * scale_down_minmax(value, min_value, max_value)) - 1


# scales down the point of gaze value
def scale_down_gaze_point(gaze_point):
	return scale_down_value(gaze_point, 0, GAZE_SCREEN_WIDTH)


# scales up all the data from 0 to 1 range or -1 to 1 range into orginal state
def scale_up_value(value, min_value, max_value):
	if SCALING == "minmax":
		return scale_up_minmax(value, min_value, max_value)
	elif SCALING == "m1p1":
		return scale_up_one_to_one(value, min_value, max_value)


# scales up all the data from 0 to 1 range into orginal state
def scale_up_minmax(value, min_value, max_value):  # min max scaling [0,1] back to original
	return (value * (max_value-min_value)) + min_value


# scales up all the data -1 to 1 range into orginal state
def scale_up_one_to_one(value, min_value, max_value):  # scaling [-1,1] back to original
	return scale_up_minmax((value+1)/2, min_value, max_value)
