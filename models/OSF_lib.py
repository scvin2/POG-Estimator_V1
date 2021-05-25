import cv2
import numpy as np
import math

def get_mean_std():
	mean = np.float32(np.array([0.485, 0.456, 0.406]))
	std = np.float32(np.array([0.229, 0.224, 0.225]))
	mean = mean / std
	std = std * 255.0
	return mean, std

def clamp_to_im(pt, w, h):
	x = pt[0]
	y = pt[1]
	if x < 0:
		x = 0
	if y < 0:
		y = 0
	if x >= w:
		x = w-1
	if y >= h:
		y = h-1
	return (int(x), int(y+1))

def logit(p, factor=16.0):
	if p >= 1.0:
		p = 0.9999999
	if p <= 0.0:
		p = 0.0000001
	p = p/(1-p)
	return float(np.log(p)) / float(factor)

def preprocess(im, crop, mean, std): # change this part for tensors
	x1, y1, x2, y2 = crop
	im = np.float32(im[y1:y2, x1:x2,::-1]) # Crop and BGR to RGB
	im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR) / std - mean
	im = np.expand_dims(im, 0)
	im = np.transpose(im, (0,3,1,2))
	return im

def preprocess_face_image(face_cords, frame, mean, std):
	res = 224.
	frame_height, frame_width = frame.shape[:2]
	(x,y,w,h) = (face_cords[0], face_cords[1], face_cords[2] - face_cords[0], face_cords[3] - face_cords[1])

	crop_x1 = x - int(w * 0.1)
	crop_y1 = y - int(h * 0.125)
	crop_x2 = x + w + int(w * 0.1)
	crop_y2 = y + h + int(h * 0.125)

	crop_x1, crop_y1 = clamp_to_im((crop_x1, crop_y1), frame_width, frame_height)
	crop_x2, crop_y2 = clamp_to_im((crop_x2, crop_y2), frame_width, frame_height)

	scale_x = float(crop_x2 - crop_x1) / res
	scale_y = float(crop_y2 - crop_y1) / res

	if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
		return [], ()

	crop = preprocess(frame, (crop_x1, crop_y1, crop_x2, crop_y2), mean, std)
	return crop, (crop_x1, crop_y1, scale_x, scale_y)

def landmarks(tensor, crop_info):
	crop_x1, crop_y1, scale_x, scale_y = crop_info
	avg_conf = 0
	lms = []
	res = 224. - 1
	for i in range(0, 66):
		m = int(tensor[i].argmax())
		x = m // 28
		y = m % 28
		conf = float(tensor[i][x,y])
		avg_conf = avg_conf + conf
		off_x = res * ((1. * logit(tensor[66 + i][x, y])) - 0.0)
		off_y = res * ((1. * logit(tensor[66 * 2 + i][x, y])) - 0.0)
		off_x = math.floor(off_x + 0.5)
		off_y = math.floor(off_y + 0.5)
		lm_x = crop_y1 + scale_y * (res * (float(x) / 27.) + off_x)
		lm_y = crop_x1 + scale_x * (res * (float(y) / 27.) + off_y)
		lms.append((lm_x,lm_y,conf))
	avg_conf = avg_conf / 66.
	lms = np.array(lms)
	lms = [np.array(lms[:,0:2])]
	return (avg_conf, lms)