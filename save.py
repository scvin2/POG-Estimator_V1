from constants import *

# function for saving extracted data into a file given its file pointer
# data can be retrived agian by using the frame id
def save_one_frame_data_to_file(frame_data, hf_file_pointer, frame_id):
	group_id = "frm-" + str(frame_id)
	group = hf_file_pointer.create_group(group_id)
	[face_img, left_eye, right_eye] = frame_data
	group.create_dataset("face_img", data=face_img)
	group.create_dataset("left_eye_img", data=left_eye)
	group.create_dataset("right_eye_img", data=right_eye)
