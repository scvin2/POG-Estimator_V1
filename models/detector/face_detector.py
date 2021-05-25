import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras import backend as K
from math import sqrt

from .fmd.fmd_detector import FMD

FILE_PATH = Path(__file__).parent.resolve()

class BaseFaceDetector():
    def __init__(self):
        pass
    
    def detect_face(self):
        raise NotImplementedError

class FMDFaceDetector(BaseFaceDetector):
    # def __init__(self, prototxt_path=FILE_PATH / "fmd" / "deploy.prototxt", weights_path=FILE_PATH / "fmd" / "res10_300x300_ssd_iter_140000.caffemodel"):
    #     self.face_detector = FMD(prototxt_path, weights_path)
    def __init__(self):
        self.face_detector = FMD()
        
    def detect_face(self, image):
        # Output bbox coordinate has ordering (y0, x0, y1, x1)
        return self.face_detector.detect_face(image)
    
class FaceAlignmentDetector(BaseFaceDetector):
    def __init__(self, fd_type="fmd"):
        self.fd_type = fd_type.lower()
        if fd_type.lower() == "fmd":
            self.fd = FMDFaceDetector()
        else:
            raise ValueError(f"Unknown face detector {face_detector}.")

        self.fd.detect_face(np.zeros((720,1280,3), np.uint8)) # dummy intialize
        self.lmd = None

    def detect_one_face(self, image):
        """
        Returns: 
            bbox_list: bboxes in [x0, y0, x1, y1] ordering (x is the horizontal axis, y the height).
        """
        bbox_list = self.fd.detect_face(image)
        image_height, image_width = image.shape[:2]
        # bbox_list = self.get_valid_bboxes(bbox_list, image_width, image_height)

        if len(bbox_list) == 0:
            return []
        if len(bbox_list) > 1:
            bbox_list = self.get_center_bbox(bbox_list, image_width, image_height)
        return bbox_list

    # @staticmethod
    # def get_valid_bboxes(bbox_list, image_width, image_height):
    #     new_bbox_list = []
    #     for bbox in bbox_list:
    #         bbox_width = bbox[2] - bbox[0]
    #         bbox_height = bbox[3] - bbox[1]
    #         if (bbox_height > (image_height/5)) and (bbox_width > (image_width/10)):
    #             new_bbox_list.append(bbox)
    #     return np.array(new_bbox_list)

    @staticmethod
    def get_center_bbox(bbox_list, image_width, image_height): # for this method add to choose based on depth image the person closet to camera is choosen
        bbox_distances = []
        image_center = [image_width/2, image_height/2]
        new_bbox_list = []
        for bbox in bbox_list:
            bbox_center = [((bbox[0] + bbox[2])/2), ((bbox[1] + bbox[3])/2)]
            dist = sqrt( (bbox_center[0] - image_center[0])**2 + (bbox_center[1] - image_center[1])**2 )
            bbox_distances.append(dist)
        center_bbox_index = np.argmin(bbox_distances)
        center_bbox = bbox_list[center_bbox_index]
        new_bbox_list.append(center_bbox)
        return np.array(new_bbox_list)