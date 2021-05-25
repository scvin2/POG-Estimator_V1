import numpy as np
import h5py
from constants import EXTRACT_DIR


# generates face and eye data for training
class DataGenerator:
	# initialize
	def __init__(self, mode, batch_size, head_motion_data_use=False):
		self.mode = mode
		self.batch_size = batch_size
		self.data_file_pointer = self.get_data_file_pointer(mode)
		self.data_combs_list = self.get_data_combs_list(hm=head_motion_data_use)
		self.data_length = len(self.data_combs_list)
		self.shuffle = self.get_shuffle(mode)
		self.epoch_steps = self.calc_epoch_steps()
		self.permute = np.array(range(self.data_length))

	# shuffles the data for training
	def get_shuffle(self, mode):
		if "test" in mode:
			return False
		else:
			return True

	# return the save file's pointer
	def get_data_file_pointer(self, mode):
		mode_file_name = mode + ".h5"
		dataset_path = EXTRACT_DIR + mode_file_name
		data_file = h5py.File(dataset_path, 'r', libver='latest')
		return data_file

	# return a array of frame ids 
	def get_data_combs_list(self, hm=False):
		if hm:
			return np.array(self.data_file_pointer.get("frame_data_combs_with_hm"))
		else:
			return np.array(self.data_file_pointer.get("frame_data_combs"))

	# calculates the number of steps in a epoch
	def calc_epoch_steps(self):
		data_length = self.data_length
		num_samples = data_length - (data_length % self.batch_size)
		return num_samples // self.batch_size

	# returns a frame id for a given reference value
	def get_data_ref_from_array(self, data_id):
		return self.data_combs_list[data_id]

	# extracts data for each from from the saved files
	def get_frame_data(self, frame_id):
		group_id = "frm-" + str(int(frame_id))
		frame_group_pointer = self.data_file_pointer.get(group_id)
		face_img = np.array(frame_group_pointer.get("face_img"))
		left_eye = np.array(frame_group_pointer.get("left_eye_img"))
		right_eye = np.array(frame_group_pointer.get("right_eye_img"))
		return [face_img, left_eye, right_eye]

	# continuosuly generates extracted data for use in training the gaze pointer model
	def generator(self):
		
		data_length = self.data_length
		max_data_length = data_length - (data_length % self.batch_size)
		
		# runs until training stops
		while True:
		
			if self.shuffle:
				self.permute = np.random.permutation(range(data_length)) 
			
			for i in range(0, max_data_length, self.batch_size):
				
				batch_face_data_t = []
				batch_left_eye_t = []
				batch_right_eye_t = []
				batch_gaze_point = []
				
				for j in range(self.batch_size):
					
					# retrive data from saved extracted_data files
					data_id = self.permute[i + j]
					data_ref = self.get_data_ref_from_array(data_id)
					[data_t_ref, gaze_point_x, gaze_point_y] = data_ref

					# gaze point on the screen for every respective camera frame
					gaze_point = [gaze_point_x, gaze_point_y]

					# facial features
					[face_data_t,  left_eye_t,  right_eye_t] = self.get_frame_data(data_t_ref)

					batch_face_data_t.append(face_data_t)
					batch_left_eye_t.append(left_eye_t)
					batch_right_eye_t.append(right_eye_t)
					batch_gaze_point.append(gaze_point)

				batch_face_data_t = np.array(batch_face_data_t, dtype=np.float32)
				batch_left_eye_t = np.array(batch_left_eye_t, dtype=np.float32)
				batch_right_eye_t = np.array(batch_right_eye_t, dtype=np.float32)
				batch_gaze_point = np.array(batch_gaze_point, dtype=np.float32)

				yield ([batch_face_data_t,  batch_left_eye_t,  batch_right_eye_t], batch_gaze_point)


if __name__ == "__main__":
	test_generator = DataGenerator("val", 80, head_motion_data_use=True)
	print(test_generator.epoch_steps)

	for value in test_generator.generator():
		print("hello")
		print(value)
		break
