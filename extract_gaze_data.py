# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import shutil
import h5py
from keras import backend as K

from models import OSF_lib
from models.detector import face_detector
from models.detector.iris_detector import IrisDetector
from participant_data import *
from images import *
from extract_lib import onnx_facial_landmarks, extract_data


def main():
	K.clear_session()

	# # tf session (limit keras gpu usage)
	# config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True  
	# config.log_device_placement = True 
	# sess = tf.Session(config=config)
	# K.tensorflow_backend.set_session(sess)

	# detector for face - retina face - dummy initialized once
	fd = face_detector.FaceAlignmentDetector(fd_type="fmd")

	# detector for eyes - mediapipe - dummy initialized once
	idet = IrisDetector()

	# detect facial landmarks - OSF_FA (OpenSeeFace-facial alignment)
	osf_fa_session = onnx_facial_landmarks()
	osf_fa_input_name = osf_fa_session.get_inputs()[0].name
	mean, std = OSF_lib.get_mean_std()
	osf_fa_values = [osf_fa_session, osf_fa_input_name, mean, std]

	print("Extracting Data....")

	# input files
	dataset_path = DATA_DIR

	# output extracted files
	extracted_data_path = EXTRACT_DIR
	if os.path.exists(extracted_data_path):
		shutil.rmtree(extracted_data_path)
	os.mkdir(extracted_data_path)

	# =================================================================================================================

	# acquire file pointers for saving extracted data

	train_data_save_path = str(extracted_data_path / "train.h5")
	val_data_save_path = str(extracted_data_path / "val.h5")
	test_data_save_path = str(extracted_data_path / "test.h5")

	hf_train = h5py.File(train_data_save_path, 'w', libver='latest')
	hf_val = h5py.File(val_data_save_path, 'w', libver='latest')
	hf_test = h5py.File(test_data_save_path, 'w', libver='latest')

	# =================================================================================================================
	# train data

	train_frames_dir_path_list, _ = get_partcipant_frame_lists(dataset_path, TRAIN_PARTICPANT_LIST)
	data_combs, id_counter_train, o_counter_train, e_counter_train = extract_data(
																					train_frames_dir_path_list, 
																					hf_train, 
																					0,
																					osf_fa_values, 
																					fd, 
																					idet
																					)

	hf_train.create_dataset("frame_data_combs", data=data_combs)

	del data_combs
	hf_train.close()

	# =================================================================================================================

	# val data

	val_dir_path_list, _ = get_partcipant_frame_lists(dataset_path, VALIDATE_PARTCIPANT_LIST)
	data_combs, id_counter_val, o_counter_val, e_counter_val = extract_data(
																				val_dir_path_list, 
																				hf_val, 
																				0,
																				osf_fa_values, 
																				fd, 
																				idet
																			)

	hf_val.create_dataset("frame_data_combs", data=data_combs)
	
	del data_combs
	hf_val.close()

	# ==================================================================================================================
	
	# test data

	test_frames_dir_path_list, _ = get_partcipant_frame_lists(dataset_path, TEST_PARTICIPANT_LIST)
	data_combs, id_counter_test, o_counter_test, e_counter_test = extract_data( 
																				test_frames_dir_path_list, 
																				hf_test, 
																				0,
																				osf_fa_values, 
																				fd, 
																				idet
																				)

	hf_test.create_dataset("frame_data_combs", data=data_combs)
	
	del data_combs
	hf_test.close()

	# ===================================================================================================================

	# extraction results

	print("train frames opened =", o_counter_train)
	print("train frames extracted =", e_counter_train)
	print("val frames opened =", o_counter_val)
	print("val frames extracted =", e_counter_val)
	print("test frames opened =", o_counter_test)
	print("test frames extracted =", e_counter_test)


if __name__ == "__main__":
	main()
