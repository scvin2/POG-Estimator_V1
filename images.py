import numpy as np
import cv2
import itertools
from constants import *
from coordinates import *
import imutils


# return a black image of for a given size parameters
def get_black_image(height, width):
	return np.zeros((height, width, 3), np.uint8)


# resizes the image to a given size parameters
def image_resize(img, width, height):
	return cv2.resize(img, (width, height))


# crops a given image symmetrically into a square shape
def square_crop_image(im, new_image_height, new_image_width):
	image_height, image_width = im.shape[:2]
	x0 = int(image_width/2-new_image_width/2)
	y0 = int(image_height/2-new_image_height/2)
	x1 = x0 + new_image_width
	y1 = y0 + new_image_height
	return im[y0:y1, x0:x1], np.array([x0, y0])


# return a heat image of given size for the given landmarks locations
def get_heat_image(img_shape, patch_size, landmarks):  # the output has elements with values between 0 to 1
	
	half_size = int(patch_size / 2)
	offsets = np.array(list(itertools.product(range(-half_size, half_size + 1), range(-half_size, half_size + 1))))
	
	# generates a landmark heatmap for a single landmark
	def draw_landmarks_helper(landmark):  # this kind of global sharing between processes will work in linux
		img = np.zeros(img_shape)
		intLandmark = landmark.astype('int32')
		locations = offsets + intLandmark
		dxdy = landmark - intLandmark
		offsetsSubPix = offsets - dxdy
		vals = 1 / (1 + np.sqrt(np.sum(offsetsSubPix * offsetsSubPix, axis=1) + 1e-6))
		img[locations[:, 1], locations[:, 0]] = vals
		return img

	landmarks[:, 0] = np.clip(landmarks[:, 0], half_size, img_shape[1] - 1 - half_size)
	landmarks[:, 1] = np.clip(landmarks[:, 1], half_size, img_shape[0] - 1 - half_size)

	# gets landmark heatmap for each landmark
	imgs = []
	for i in landmarks:
		imgs.append(draw_landmarks_helper(i))

	# combines all the heatmaps into a single heatmap image and returns its
	imgs = np.array(imgs)
	return imgs.max(axis=0)


# utilizes the face coordrinates for croping the face from the color and depth images 
# face heat map is generated using the landmarks
# returns the face color, depth and heat maps images
def get_face_cdh_images(color_img, depth_img, face_cords, landmarks_for_depth_img):

	# make the coordinates of the face image to  form a square shape
	face_cords_symmetric = make_coords_symmetric(face_cords, depth_img)
	if face_cords_symmetric is None:
		print("couldn't get face cdh")
		return None, None, None
	[x0, y0, x1, y1] = face_cords_symmetric
	
	new_origin = np.array([x0, y0])
	new_landmarks = landmarks_for_depth_img - new_origin

	face_color_img = color_img[y0:(y1+1), x0:(x1+1)]
	face_depth_img = depth_img[y0:(y1+1), x0:(x1+1)]
	old_shape = face_depth_img.shape
	
	# trasnform the face image for neural network input
	face_color_img = cv2.cvtColor(face_color_img, cv2.COLOR_BGR2GRAY)
	face_color_img = cv2.resize(face_color_img, FACE_DEPTH_IMG_CROP_SHAPE, interpolation=cv2.INTER_AREA)
	face_color_img = cv2.equalizeHist(face_color_img)

	# transform the face depth image for neural network input
	face_depth_img = cv2.resize(face_depth_img, FACE_DEPTH_IMG_CROP_SHAPE, interpolation=cv2.INTER_NEAREST)
	face_depth_img = np.clip(face_depth_img, 0, MAX_CAPTURE_DISTANCE)
	new_shape = face_depth_img.shape

	# transform the landmarks coordinates from full image to the face image 
	resize_shift = np.array(new_shape)/np.array(old_shape)
	new_landmarks = new_landmarks * resize_shift
	new_landmarks = (np.round(new_landmarks)).astype('int16')
	
	# generate the face heat map image
	# the output has elements with values between 0 to 1
	face_heat_img = get_heat_image(FACE_DEPTH_IMG_CROP_SHAPE, 16, new_landmarks)
	
	return face_color_img, face_depth_img, face_heat_img


# crops and return an eye depth and color image
# also return a heatmap for the eye image by using the given eye landmarks coordinates
# all eye points -> (x,y)
def get_eye_gdh_images(eye_edge_l, eye_edge_r, outer_eye_lms, all_eye_lms, color_img, depth_img):
	
	# calculate average distance of eye from the camera
	eye_center = np.mean(outer_eye_lms, axis=0)  # (x,y)
	distance = np.mean(depth_img[
									(int(eye_center[1])-2):(int(eye_center[1])+3),
									(int(eye_center[0])-2):(int(eye_center[0])+3)
								]
						)
	if distance < MIN_DEPTH_ERROR_DISTANCE:
		print("couldn obtain eye gdh")
		return None, None, None
	
	# calcule the rotation angle of the eye in the image
	# theta = atan2((y1 - y0), (x1 - x0)) 
	rotate_angle = round((180/math.pi) * math.atan2(eye_edge_r[1]-eye_edge_l[1], eye_edge_r[0]-eye_edge_l[0]))
	
	focal_x = CAMERA_FOCAL_X
	focal_y = CAMERA_FOCAL_Y
	rx = 70  # arbitrarily choosen for the lab setup
	ry = 70  # arbitrarily choosen for the lab setup
	wx = (focal_x * rx)/distance 
	wy = (focal_y * ry)/distance

	x0 = int(eye_center[0] - (wx/2))
	x1 = int(round(x0 + wx))
	y0 = int(eye_center[1] - (wy/2))
	y1 = int(round(y0 + wy))

	# crop the eye image from the full images
	eye_image_large_color = color_img[y0:(y1+1), x0:(x1+1)]
	eye_image_large_depth = depth_img[y0:(y1+1), x0:(x1+1)]

	# transform the eye landmark coordinates to the eye image
	eye_landmarks = all_eye_lms - np.array([x0, y0])

	# convert the eye image to gray
	eye_image_large_color = cv2.cvtColor(eye_image_large_color, cv2.COLOR_BGR2GRAY)

	# rotate the eye images for stabillization
	eye_image_large_color_rotated = imutils.rotate(eye_image_large_color, angle=rotate_angle)
	eye_image_large_depth_rotated = imutils.rotate(eye_image_large_depth, angle=rotate_angle)

	image_height, image_width = eye_image_large_color.shape[:2]
	eye_landmarks = np.array([rotate_point_for_rotated_image(rotate_angle, image_width, image_height, mark[0], mark[1]) for mark in eye_landmarks])

	# for cropping and using only part of the rotated images
	new_image_height = int((image_height * 2)/3)
	new_image_width = int((image_width * 2)/3)

	# make the images symmetric and shift the landmarks 
	eye_color_img, new_eye_x0y0 = square_crop_image(eye_image_large_color_rotated, new_image_height, new_image_width)
	eye_depth_img, _ = square_crop_image(eye_image_large_depth_rotated, new_image_height, new_image_width)
	eye_landmarks = eye_landmarks - new_eye_x0y0
	old_shape = eye_color_img.shape

	# reshape and transform images for the neural network input
	eye_color_img = cv2.resize(eye_color_img, EYE_DEPTH_IMG_CROP_SHAPE, interpolation=cv2.INTER_AREA)
	eye_color_img = cv2.equalizeHist(eye_color_img)
	eye_depth_img = cv2.resize(eye_depth_img, EYE_DEPTH_IMG_CROP_SHAPE, interpolation=cv2.INTER_NEAREST)
	eye_depth_img = np.clip(eye_depth_img, 0, MAX_CAPTURE_DISTANCE)
	
	# transform the landmarks to the new image shape
	new_shape = eye_color_img.shape
	resize_shift = np.array(new_shape)/np.array(old_shape)
	eye_landmarks = eye_landmarks * resize_shift
	eye_landmarks = (np.round(eye_landmarks)).astype('int16')
	
	# generate eye heatmap image for the eye landmark coordinates 
	eye_heat_img = get_heat_image(EYE_DEPTH_IMG_CROP_SHAPE, 13, eye_landmarks)
	
	return eye_color_img, eye_depth_img, eye_heat_img
