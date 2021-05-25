from pathlib import Path
import numpy as np
import cv2
import os
import pyximport
pyximport.install()

from .retinaface_cov import RetinaFaceCoV

class FMD():
	def __init__(self):
		FILE_PATH = Path(__file__).parent.resolve()
		model_path = str(FILE_PATH / "mnet_cov2")
		self.thresh = 0.85
		self.mask_thresh = 0.2
		self.scales = [640, 1080]
		self.faceNet = RetinaFaceCoV(model_path, 0, -1, 'net3l')
    
	def detect_face(self, image):
		bboxlist = self.detect_and_predict_mask(image, self.faceNet)
		return bboxlist

	def detect_and_predict_mask(self, frame, faceNet):
		im_shape = frame.shape
		(h, w) = im_shape[:2]
		target_size = self.scales[0]
		max_size = self.scales[1]
		im_size_min = np.min(im_shape[0:2])
		im_size_max = np.max(im_shape[0:2])
		#if im_size_min>target_size or im_size_max>max_size:
		im_scale = float(target_size) / float(im_size_min)
		# prevent bigger axis from being more than max_size:
		if np.round(im_scale * im_size_max) > max_size:
			im_scale = float(max_size) / float(im_size_max)
		scale = [im_scale]
		flip = False

		detections, _ = faceNet.detect(frame,self.thresh,scales=scale,do_flip=flip)

		locs = []
		for i in range(detections.shape[0]):
			face = detections[i]
			box = face[0:4].astype(np.int)
			mask = face[5]
			mask_on = 0
			if mask >= self.mask_thresh:
				mask_on = 1

			(startX, startY, endX, endY) = box
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			locs.append([startX, startY, endX, endY, mask_on])
		locs = np.array(locs)
		if 0 == len(locs):
			locs = np.zeros((1, 5))
		return locs

	