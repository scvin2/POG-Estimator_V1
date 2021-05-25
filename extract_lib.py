import onnxruntime
import os
from models import OSF_lib
from images import *
import numpy as np
from constants import *
from scaling import scale_down_gaze_point
import cv2
from preprocess import scale_down_preprocess_data_for_training
from save import save_one_frame_data_to_file


# onnx configuration for facial landmarks
# can run directly on the cpu
def onnx_facial_landmarks():
	# detect facial landmarks - OSF_FA (OpenSeeFace-facial alignment)
	options = onnxruntime.SessionOptions()
	options.inter_op_num_threads = 1
	options.intra_op_num_threads = 2  # max threads
	options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
	options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
	options.log_severity_level = 3

	model_base_path = os.path.join(os.path.dirname(__file__), os.path.join("weights"))
	osf_fa_model_name = "mnv3_opt_b.onnx"
	osf_fa_session = onnxruntime.InferenceSession(os.path.join(model_base_path, osf_fa_model_name), sess_options=options)

	return osf_fa_session


# extracts eye mark coordinates along with modified face landmarks with new eye landmarks 
def get_eye_landmark_coords(face_cords, color_img, osf_fa_values, idet):
	[osf_fa_session, osf_fa_input_name, mean, std] = osf_fa_values

	crop, crop_info = OSF_lib.preprocess_face_image(face_cords, color_img, mean, std)
	if len(crop_info) == 0:
		print("couldn't get face landmarks")
		return None

	# get facial landmark coordinates
	face_landmarks_model_out = osf_fa_session.run([], {osf_fa_input_name: crop})[0]
	conf, lms = OSF_lib.landmarks(face_landmarks_model_out[0], crop_info)

	# obtains all eye landmarks but use only outer eye lid and iris landmarks
	eye_landmarks = idet.detect_iris(color_img, landmarks=lms)
	left_eye_pts = eye_landmarks[0]  # here each landmark is in [x,y] format
	right_eye_pts = eye_landmarks[1]  # here each landmark is in [x,y] format

	# elg - data from eye landmark neuural net
	left_eye_iris = left_eye_pts[8:(len(left_eye_pts)-1)]
	right_eye_iris = right_eye_pts[8:(len(right_eye_pts)-1)]

	# direct face eyelandmarks
	left_eye_idx = slice(36, 42)
	right_eye_idx = slice(42, 48)
	lm = lms[0]
	left_eye = np.array(lm[left_eye_idx])[:, ::-1]
	right_eye = np.array(lm[right_eye_idx])[:, ::-1]

	left_eye = np.concatenate([left_eye, left_eye_iris], axis=0)
	right_eye = np.concatenate([right_eye, right_eye_iris], axis=0)

	left_eye = expand_eye_points(left_eye)
	right_eye = expand_eye_points(right_eye)

	left_eye_for_depth_img = left_eye
	right_eye_for_depth_img = right_eye

	# remove less detialed eye landmarks from face landmarks
	lms_for_depth = lms[0]
	lms_for_depth = lms_for_depth[:, ::-1]
	lms_for_depth = np.concatenate([lms_for_depth[:17, :], lms_for_depth[27:36, :], lms_for_depth[48:, :]], axis=0)

	# combine face landmarks with new detailed eye landmarks
	landmarks_for_depth_img = np.concatenate([lms_for_depth, left_eye_for_depth_img, right_eye_for_depth_img], axis=0)
	
	return (left_eye, right_eye, landmarks_for_depth_img)


# extracts face, left eye and right eye features from the color and depth images
def get_facial_features(color_img, depth_img, osf_fa_values, fd, idet):
	# get face coords
	face = fd.detect_one_face(color_img)
	if len(face) == 0:
		print("couldn't find face")
		return None
	face_cords = face[0][0:4]  # x0, y0, x1, y1
	face_cords = [int(i) for i in face_cords]  # x0, y0, x1, y1

	# get eye landmarks
	eyes = get_eye_landmark_coords(face_cords, color_img, osf_fa_values, idet)
	if eyes is None:
		return None
	else:
		(left_eye, right_eye, landmarks_for_depth_img) = eyes

	# the output heatmap has elements with values between 0 to 1
	# the output depth image has elements with values between 0 to 2000
	face_color_img, face_depth_img, face_heat_img = get_face_cdh_images(
																			color_img,
																			depth_img,
																			face_cords,
																			landmarks_for_depth_img
																		)
	if face_color_img is None:
		return None

	# left eye images
	left_eye_color_img, left_eye_depth_img, left_eye_heat_img = get_eye_gdh_images(
																					left_eye[0], 
																					left_eye[8], 
																					left_eye[0:16], 
																					left_eye, 
																					color_img, 
																					depth_img
																					)

	if left_eye_color_img is None:
		return None

	# right eye images
	right_eye_color_img, right_eye_depth_img, right_eye_heat_img = get_eye_gdh_images(
																						right_eye[0], 
																						right_eye[8], 
																						right_eye[0:16], 
																						right_eye, 
																						color_img, 
																						depth_img
																						)
	if right_eye_color_img is None:
		return None

	output_images = [
						[face_color_img, face_depth_img, face_heat_img],
						[left_eye_color_img, left_eye_depth_img, left_eye_heat_img],
						[right_eye_color_img, right_eye_depth_img, right_eye_heat_img]
					]

	return output_images


# expand eye landmarks with averages of between detected eye landmarks
def expand_eye_points(eye):
	marks = eye.tolist()
	up_point = (eye[1] + eye[2]) / 2
	down_point = (eye[4] + eye[5]) / 2
	marks.append(up_point)
	marks.append(down_point)
	return np.array(marks)


# extracts all facial data from the data path one by one and saves them to a h5py file
def extract_data(participants_frames_dir_path_list, hf_file_pointer, start_id, osf_fa_values, fd, idet):
	# counters
	e_counter = 0  # no of extracted data
	o_counter = 0  # no of opened data
	id_counter = start_id

	# output
	combs_list = []

	# goes by each frame data directory
	# each directory will have color and depth images and gaze point data for number of recorded continous frames
	for frame_dir in participants_frames_dir_path_list:
		
		gaze_point_path = str(frame_dir / "gaze_point.npy")
		gaze_point = scale_down_gaze_point(np.load(gaze_point_path))

		total_frames_before = id_counter

		for n in range(NO_OF_FRAMES_PER_MARK):
			
			o_counter += 1
			color_img_name = "color_n" + str(n) + ".png"
			color_img_path = str(frame_dir / color_img_name)
			depth_img_name = "transformed_depth_n" + str(n) + ".npy"
			depth_img_path = str(frame_dir / depth_img_name)

			color_img = cv2.imread(color_img_path)
			depth_img = np.load(depth_img_path)

			# obtain features of face and detailed eyes
			data = get_facial_features(color_img, depth_img, osf_fa_values, fd, idet)
			if not data:
				print(frame_dir)
				continue
			data = scale_down_preprocess_data_for_training(data)

			# save extracted frame to file
			save_one_frame_data_to_file(data, hf_file_pointer, id_counter)

			id_counter += 1
			e_counter += 1

		total_frames_after = id_counter
		# newly_added_frames_total =  total_frames_after - total_frames_before

		# data is saved here in "[t,gaze_point_index]" format  # here 't' frames are only reference indexs to frame array
		n_data = [[i, gaze_point[0], gaze_point[1]] for i in range(total_frames_before, total_frames_after)]
		combs_list += n_data

		# hf_file_pointer.flush()

	return np.array(combs_list), id_counter, o_counter, e_counter
