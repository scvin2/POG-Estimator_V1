import math
import numpy as np
from constants import *


# returns the new coordinates (x,y) in the image after being rotated by given angle
# utilizes rotation matrix for calculating the coordinates directly
def rotate_point_for_rotated_image(angle, image_width, image_height, x, y):
	theta = (angle * math.pi) / 180
	h = image_height - 1
	x0 = round((image_width-1)/2)
	y0 = round(h/2)
	x_new = ((x-x0)*math.cos(theta)) - ((h-y-y0)*math.sin(theta)) + x0
	y_new = (-(x-x0)*math.sin(theta)) - ((h-y-y0)*math.cos(theta)) + (h-y0)
	return [x_new, y_new]


# works only if no holes in the depth image
# need inpainting of depth image if holes present
# calculates the average depth value at a given coordinate in depth image
def get_avg_face_depth(center, depth_img):
	c_x = int(center[0])
	c_y = int(center[1])
	center_pixels = depth_img[(c_y-2):(c_y+3), (c_x-2):(c_x+3)]
	return np.mean(center_pixels)


# ###---- Inefficient need to change this ##############################################################
# adjusts the image size to symmetric
def make_coords_symmetric(face_cords, depth_img):
	[x0, y0, x1, y1] = face_cords

	face_center = [(x0+x1)/2, (y0+y1)/2]
	face_distance = get_avg_face_depth(face_center, depth_img)
	
	if face_distance < MIN_DEPTH_ERROR_DISTANCE:
		return None

	focal_x = CAMERA_FOCAL_X
	focal_y = CAMERA_FOCAL_Y
	rx = 210  # value arbitrarily choosen
	ry = 210  # value arbitrarily choosen
	wx = (focal_x * rx)/face_distance
	wy = (focal_y * ry)/face_distance
	x0 = int(face_center[0] - (wx/2))
	x1 = round(x0 + wx)
	y0 = int(face_center[1] - (wy/2))
	y1 = round(y0 + wy)
	x_width = x1-x0
	y_width = y1-y0
	
	if y_width > x_width:
		yx_diff = y_width - x_width
		add_yx_diff = int(yx_diff/2)
		diff = yx_diff % 2
		x0 = x0 - add_yx_diff 
		x1 = x1 + add_yx_diff + diff
	elif x_width > y_width:
		xy_diff = x_width - y_width
		add_xy_diff = int(xy_diff/2)
		diff = xy_diff % 2
		y0 = y0 - add_xy_diff 
		y1 = y1 + add_xy_diff + diff

	if x0 < 0:
		x1 += abs(x0)
		x0 = 0
	elif x1 >= CAPTURE_IMAGE_WIDTH:
		x0 = x0 - (x1-(CAPTURE_IMAGE_WIDTH-1))
		x1 = CAPTURE_IMAGE_WIDTH-1

	if y0 < 0:
		y1 += abs(y0)
		y0 = 0
	elif y1 >= CAPTURE_IMAGE_HEIGHT:
		y0 = y0 - (y1-(CAPTURE_IMAGE_HEIGHT-1))
		y1 = CAPTURE_IMAGE_WIDTH-1

	return [int(x0), int(y0), int(x1), int(y1)]
