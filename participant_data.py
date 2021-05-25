import numpy as np
import os


# return praticipant data lists for extraction
def get_partcipant_frame_lists(data_path, participant_list):
	participants_all_frames_dir_path_list = []
	participant_frames_both_hm_list = []

	for participant_dir in participant_list:

		participant_dir_path = data_path / participant_dir
		participant_eyes_dir_path = participant_dir_path / "eyes_only"
		participant_head_dir_path = participant_dir_path / "head_only"
		participant_both_dir_path = participant_dir_path / "both"

		participants_all_frames_dir_path_list += (
													list_dir_paths(participant_eyes_dir_path) + 
													list_dir_paths(participant_head_dir_path) + 
													list_dir_paths(participant_both_dir_path)
												)  # posix paths need to change to str

	return np.array(participants_all_frames_dir_path_list), np.array(participant_frames_both_hm_list)


# returns a list of all file paths in a given directory
def list_dir_paths(dir_path):
	dir_list = os.listdir(dir_path)
	dir_paths_list = []
	for x in dir_list:
		path = dir_path / x
		dir_paths_list.append(path)
	return dir_paths_list
