from scaling import *
from constants import *
import numpy as np


# scales down all the face, left eye and right eye data and gives an output stacked images data
def scale_down_preprocess_data_for_training(data):
	[face, left_eye, right_eye] = data
	[face_color_img, face_depth_img, face_heat_img] = face
	[left_eye_color_img, left_eye_depth_img, left_eye_heat_img] = left_eye
	[right_eye_color_img, right_eye_depth_img, right_eye_heat_img] = right_eye

	# scaling down face image data
	face_color_img = scale_down_value(face_color_img, 0, 255)  # image is in grayscale format
	face_depth_img = scale_down_value(face_depth_img, 0, MAX_CAPTURE_DISTANCE)
	face_heat_img = scale_down_value(face_heat_img, 0, 1)
	
	# scaling down left eye image data
	left_eye_color_img = scale_down_value(left_eye_color_img, 0, 255)  # image is in grayscale format
	left_eye_depth_img = scale_down_value(left_eye_depth_img, 0, MAX_CAPTURE_DISTANCE)
	left_eye_heat_img = scale_down_value(left_eye_heat_img, 0, 1)

	# scaling down right image data
	right_eye_color_img = scale_down_value(right_eye_color_img, 0, 255)  # image is in grayscale format
	right_eye_depth_img = scale_down_value(right_eye_depth_img, 0, MAX_CAPTURE_DISTANCE)
	right_eye_heat_img = scale_down_value(right_eye_heat_img, 0, 1)

	# stacking data for neural network input
	face_img = np.dstack((face_color_img, face_depth_img, face_heat_img))
	left_eye_img = np.dstack((left_eye_color_img, left_eye_depth_img, left_eye_heat_img))
	right_eye_img = np.dstack((right_eye_color_img, right_eye_depth_img, right_eye_heat_img))
	
	return [face_img, left_eye_img, right_eye_img]


# scales down all the face, left eye and right eye data and gives an output stacked images data - FOR TESTING
def preprocess_input_data(input_data):
	[face, left_eye, right_eye] = input_data
	[face_color_img, face_depth_img, face_heat_img] = face
	[left_eye_color_img, left_eye_depth_img, left_eye_heat_img] = left_eye
	[right_eye_color_img, right_eye_depth_img, right_eye_heat_img] = right_eye

	# scaling down face image data
	face_color_img = scale_down_value(face_color_img, 0, 255)  # image is in grayscale format
	face_depth_img = scale_down_value(face_depth_img, 0, MAX_CAPTURE_DISTANCE)
	face_heat_img = scale_down_value(face_heat_img, 0, 1)
	
	# scaling down left eye image data
	left_eye_color_img = scale_down_value(left_eye_color_img, 0, 255)  # image is in grayscale format
	left_eye_depth_img = scale_down_value(left_eye_depth_img, 0, MAX_CAPTURE_DISTANCE)
	left_eye_heat_img = scale_down_value(left_eye_heat_img, 0, 1)

	# scaling down right image data
	right_eye_color_img = scale_down_value(right_eye_color_img, 0, 255)  # image is in grayscale format
	right_eye_depth_img = scale_down_value(right_eye_depth_img, 0, MAX_CAPTURE_DISTANCE)
	right_eye_heat_img = scale_down_value(right_eye_heat_img, 0, 1)

	# stacking data for neural network input
	face_img = np.dstack((face_depth_img, face_heat_img))
	left_eye_img = np.dstack((left_eye_color_img, left_eye_depth_img, left_eye_heat_img))
	right_eye_img = np.dstack((right_eye_color_img, right_eye_depth_img, right_eye_heat_img))

	face_img = np.expand_dims(face_img, 0)
	left_eye_img = np.expand_dims(left_eye_img, 0)
	right_eye_img = np.expand_dims(right_eye_img, 0)
	
	return [face_img, left_eye_img, right_eye_img]


# convert scaled output from neural network into real screen coordinates 
def modify_data_from_gp_output(output_data):
	gp_data = output_data[0]
	gp_data = [
				scale_up_value(gp_data[0], 0, MAX_GAZE_SCREEN_WIDTH_VAL),
				scale_up_value(gp_data[1], 0, MAX_GAZE_SCREEN_HEIGHT_VAL)
				]
	
	# fix 
	gp_data = [round(gp_data[0]), round(gp_data[1])]
	gp_data[0] = max(gp_data[0], 0)
	gp_data[1] = max(gp_data[1], 0)
	gp_data[0] = min(gp_data[0], MAX_GAZE_SCREEN_WIDTH_VAL)
	gp_data[1] = min(gp_data[1], MAX_GAZE_SCREEN_HEIGHT_VAL)
	
	return gp_data
