import time
import os
import numpy as np
import cv2
import imutils
from keras import backend as K
from keras.models import load_model

from models.detector import face_detector
from models.detector.iris_detector import IrisDetector
from models import OSF_lib
from constants import *
from scaling import *
import preprocess
from coordinates import *
from loss import euclidean_loss
from images import *
from pointer import *
from camera import configure_camera
from extract_lib import onnx_facial_landmarks, get_facial_features


def main():

	K.clear_session()

	# # tf session (limit keras gpu usage)
	# config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True  
	# config.log_device_placement = True 
	# sess = tf.Session(config=config)
	# K.tensorflow_backend.set_session(sess)

	# detect face - fmd (face mask detector) - dummy initialized once
	fd = face_detector.FaceAlignmentDetector(fd_type="fmd")

	# decect eyes - ELG - dummy initialized once
	idet = IrisDetector() 

	# detect facial landmarks - OSF_FA (OpenSeeFace-facial alignment)
	osf_fa_session = onnx_facial_landmarks()
	osf_fa_input_name = osf_fa_session.get_inputs()[0].name
	mean, std = OSF_lib.get_mean_std()
	osf_fa_values = [osf_fa_session, osf_fa_input_name, mean, std]

	# gaze pointer
	model_path = str(FILE_PATH / "gaze_pointer_checkpoints" / "weights-LR0.001.E00019-L0.1627-VL0.2096-VA0.849.hdf5")
	gp_model = load_model(model_path, custom_objects={'euclidean_loss': euclidean_loss})
	# dummy_input = [
	# 				np.expand_dims(np.zeros(FACE_DATA_SHAPE, np.float32), 0),
	# 				np.expand_dims(np.zeros(EYE_DATA_SHAPE, np.float32), 0),
	# 				np.expand_dims(np.zeros(EYE_DATA_SHAPE, np.float32), 0)
	# 				]
	# dummy_output = gp_model.predict(dummy_input) #  dummy initialize gaze pointer once

	# all circle points list
	# circle_center_pts_list = get_pts_list()

	# camera
	k4a = configure_camera()
	k4a.start()
	time.sleep(2.0)

	# cv2 set fullscreen
	cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
	cv2.setWindowProperty('result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	# video capture setup
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # (*'MP42')
	video_output_path = str(FILE_PATH / "output.avi")
	video_output = cv2.VideoWriter(video_output_path, fourcc, 10.0, (VIDEO_OUTPUT_FRAME_WIDTH, VIDEO_OUTPUT_FRAME_HEIGHT))

	# initialize pointer position
	# gaze pointer starts from center of screen
	current_pointer_position = [round(GAZE_SCREEN_WIDTH/2), round(GAZE_SCREEN_HEIGHT/2)]

	# features list input queue, always will have 3 elements, [t-2, t-1, t]
	# dummy_features = [
	# 					np.expand_dims(np.zeros(FACE_DATA_SHAPE, np.float32), 0),
	# 					np.expand_dims(np.zeros(EYE_DATA_SHAPE, np.float32), 0),
	# 					np.expand_dims(np.zeros(EYE_DATA_SHAPE, np.float32), 0)
	# 					]
	# features_queue = [dummy_features, dummy_features, dummy_features]

	while True:

		# det a RGBD frame from the camera
		while True:
			capture = k4a.get_capture()
			if np.any(capture.depth):
				break
		
		# capture RGBD frame
		color_img = capture.color
		depth_img = capture.transformed_depth

		# change image to opencv image
		color_img = color_img[:, :, :3]  # BGRA to BGR

		# extract data
		features_data = get_facial_features(color_img, depth_img, osf_fa_values, fd, idet)
		if not features_data:
			continue

		if features_data[0][0] is None:
			continue
		if features_data[1][0] is None:
			continue
		if features_data[2][0] is None:
			continue

		gaze_pointer_input_t = preprocess.preprocess_input_data(features_data) 
		gaze_pointer_output = gp_model.predict(gaze_pointer_input_t)
		
		# ***********************OUTPUT VALUE - GP **********************
		gaze_pointer_output = preprocess.modify_data_from_gp_output(gaze_pointer_output) 

		# print (gaze_pointer_output)
		current_pointer_position = update_cursor_pointer(POINTER_RADIUS, current_pointer_position, gaze_pointer_output)

		# video output frames
		video_frame = get_black_image(VIDEO_OUTPUT_FRAME_HEIGHT, VIDEO_OUTPUT_FRAME_WIDTH)
		# face_frame = color_img[face_cords[1]:face_cords[3], face_cords[0]:face_cords[2]]
		# face_frame = image_resize(face_frame, 300, 400)
		# video_frame[670:(670+400), 0:(0+300), :] = face_frame[:, :, :]  # overlay face image over video frame

		# test circles
		video_frame = cv2.circle(video_frame, (480, 270), 100, (0, 69, 255), -1)  # left
		video_frame = cv2.circle(video_frame, (1440, 270), 100, (0, 69, 255), -1)  # right
		video_frame = cv2.circle(video_frame, (960, 850), 100, (0, 69, 255), -1)  # center down

		# here, cv2.circle(image, center_coordinates, radius, color, thickness)
		video_frame = cv2.circle(
									video_frame,
									(current_pointer_position[0], current_pointer_position[1]),
									POINTER_SHOW_RADIUS,
									(255, 0, 0),
									10
									)

		# resize frames for output
		output_frame = image_resize(video_frame, VIDEO_OUTPUT_FRAME_WIDTH, VIDEO_OUTPUT_FRAME_HEIGHT)
		video_output.write(output_frame)

		cv2.imshow('result', video_frame)
		if (cv2.waitKey(1) & 0xFF) == ord('q'):
			break

	k4a.stop()
	video_output.release()
	cv2.destroyAllWindows()

	print("Closed")


if __name__ == '__main__':
	main()
