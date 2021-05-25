import numpy as np
import cv2
from pathlib import Path
import imutils
import math

from .ELG.elg_keras import KerasELG

FILE_PATH = str(Path(__file__).parent.resolve())
NET_INPUT_SHAPE = (108, 180)

class IrisDetector():
    def __init__(self, path_elg_weights=FILE_PATH+"/ELG/elg_keras.h5"):
        self.detector = None
        self.elg = KerasELG()
        self.elg.net.load_weights(path_elg_weights)
        self.elg.net.predict(np.concatenate([np.random.rand(1,108,180,1), np.random.rand(1,108,180,1)], axis=0)) # dummy intialize
        
    def set_detector(self, detector):
        self.detector = detector
        
    def detect_iris(self, im, landmarks=None, rotate_allow=True):
        """
        Input:
            im: RGB image
        Outputs:
            output_eye_landmarks: list of eye landmarks having shape (2, 18, 2) with ordering (L/R, landmarks, x/y).
        """
            
        if landmarks == None:
            try:    
                faces, landmarks = self.detector.detect_face(im, with_landmarks=True)     
            except:
                raise NameError("Error occured during face detection. Maybe face detector has not been set.")
     
        left_eye_idx = slice(36, 42)
        right_eye_idx = slice(42, 48)
        # output_eye_landmarks = []
        for lm in landmarks:

            #face crop
            face, face_x0y0 = self.get_face_roi(im, lm) # face_xy -> (y,x)
            lm = (np.array(lm) - np.array(face_x0y0)).tolist()
            left_eye_ldms = lm[left_eye_idx]
            right_eye_ldms = lm[right_eye_idx]

            full_img_left_eye = face
            full_img_right_eye = face

            if rotate_allow == True:
                left_eye_mark_a = lm[36] # (y,x)
                left_eye_mark_b = lm[39] # (y,x)
                right_eye_mark_a = lm[42] # (y,x)
                right_eye_mark_b = lm[45] # (y,x)
                left_eye_angle = round((180/math.pi) * math.atan2(left_eye_mark_b[0]-left_eye_mark_a[0],left_eye_mark_b[1]-left_eye_mark_a[1]))   # theta = atan2((y1 - y0), (x1 - x0)) 
                right_eye_angle = round((180/math.pi) * math.atan2(right_eye_mark_b[0]-right_eye_mark_a[0],right_eye_mark_b[1]-right_eye_mark_a[1]))
                # print(str(left_eye_angle) + "......" + str(right_eye_angle) )
                full_img_left_eye = imutils.rotate(face, angle=left_eye_angle)
                full_img_right_eye = imutils.rotate(face, angle=right_eye_angle) # check for faster rotate function
                # full_img_left_eye = self.rotate_image(im, left_eye_angle)
                # full_img_right_eye = self.rotate_image(im, right_eye_angle)
                image_height, image_width = face.shape[:2]
                left_eye_ldms = [self.rotate_point_for_rotated_image(left_eye_angle, image_width, image_height, mark[1], mark[0]) for mark in left_eye_ldms]
                right_eye_ldms = [self.rotate_point_for_rotated_image(right_eye_angle, image_width, image_height, mark[1], mark[0]) for mark in right_eye_ldms]



            left_eye_im, left_x0y0 = self.get_eye_roi(full_img_left_eye, left_eye_ldms)
            right_eye_im, right_x0y0 = self.get_eye_roi(full_img_right_eye, right_eye_ldms)
            inp_left = self.preprocess_eye_im(left_eye_im)
            inp_right = self.preprocess_eye_im(right_eye_im)
            
            input_array = np.concatenate([inp_left, inp_right], axis=0)            
            pred_left, pred_right = self.elg.net.predict(input_array)
            
            lms_left = self.elg._calculate_landmarks(pred_left, eye_roi=left_eye_im)
            lms_right = self.elg._calculate_landmarks(pred_right, eye_roi=right_eye_im)

            lms_left += np.array(left_x0y0).reshape(1,1,2)
            lms_right += np.array(right_x0y0).reshape(1,1,2)

            if rotate_allow == True:
                lms_left = lms_left[0,:,:]
                lms_right = lms_right[0,:,:]
                lms_left = [self.rotate_point_for_rotated_image(-left_eye_angle, image_width, image_height, mark[1], mark[0]) for mark in lms_left]
                lms_right = [self.rotate_point_for_rotated_image(-right_eye_angle, image_width, image_height, mark[1], mark[0]) for mark in lms_right]
                lms_left = np.array([lms_left])
                lms_right = np.array([lms_right])

            lms_left += np.array([face_x0y0]).reshape(1,1,2)
            lms_right += np.array([face_x0y0]).reshape(1,1,2)


            lms_left = np.flip(lms_left, 2)
            lms_right = np.flip(lms_right, 2)

            eye_landmarks = np.concatenate([lms_left, lms_right], axis=0)
            # eye_landmarks = eye_landmarks + np.array([left_x0y0, right_x0y0]).reshape(2,1,2)
            # output_eye_landmarks.append(eye_landmarks)

            # cv2.imshow('left_eye', inp_left[0])
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # cv2.imshow('right_eye', inp_right[0])
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # left_eye_im1 = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
            # right_eye_im1 = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
            # cv2.imshow('left_eye1', left_eye_im1)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # cv2.imshow('right_eye1', right_eye_im1)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # return output_eye_landmarks
        return eye_landmarks

    @staticmethod
    def rotate_point_for_rotated_image(angle, image_width, image_height, x, y):
        theta = (angle * math.pi) / 180
        h = image_height - 1
        x0 = round((image_width-1)/2)
        y0 = round(h/2)
        xNew = ((x-x0)*math.cos(theta)) - ((h-y-y0)*math.sin(theta)) + x0
        yNew = (-(x-x0)*math.sin(theta)) - ((h-y-y0)*math.cos(theta)) + (h-y0)
        return [yNew, xNew]

    @staticmethod
    def get_face_roi(im, lms):
        h, w = im.shape[:2]
        min_xy = np.min(lms, axis=0)
        max_xy = np.max(lms, axis=0)
        min_xy = min_xy.tolist()
        max_xy = max_xy.tolist()
        min_xy[0] = round(min_xy[0]) - 50
        max_xy[0] = round(max_xy[0]) + 50
        min_xy[1] = round(min_xy[1]) - 50
        max_xy[1] = round(max_xy[1]) + 50
        min_xy[0] = max(min_xy[0], 0)
        min_xy[1] = max(min_xy[1], 0)
        max_xy[0] = min(h, max_xy[0])
        max_xy[1] = min(w, max_xy[1])
        face = im[min_xy[0]:max_xy[0], min_xy[1]:max_xy[1]]
        return face, min_xy
    
    @staticmethod
    def get_eye_roi(im, lms, ratio_w=1.5):
        def adjust_hw(hw, ratio_w=1.5):
            """
            set RoI height and width to the same ratio of NET_INPUT_SHAPE
            """
            h, w = hw[0], hw[1]
            new_w = w * ratio_w
            new_h = NET_INPUT_SHAPE[0] / NET_INPUT_SHAPE[1] * new_w
            return np.array([new_h, new_w])
        h, w = im.shape[:2]
        min_xy = np.min(lms, axis=0)
        max_xy = np.max(lms, axis=0)
        hw = max_xy - min_xy
        hw = adjust_hw(hw, ratio_w=ratio_w)
        center = np.mean(lms, axis=0)
        x0, y0 = center - (hw) / 2
        x1, y1 = center + (hw) / 2
        x0, y0, x1, y1 = map(np.int32,[x0, y0, x1, y1])
        x0, y0 = np.maximum(x0, 0), np.maximum(y0, 0)
        x1, y1 = np.minimum(x1, h), np.minimum(y1, w)
        eye_im = im[x0:x1, y0:y1]
        return eye_im, (x0, y0)
    
    @staticmethod
    def preprocess_eye_im(im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.equalizeHist(im)
        im = cv2.resize(im, (NET_INPUT_SHAPE[1], NET_INPUT_SHAPE[0]))[np.newaxis, ..., np.newaxis]
        im = im / 255 * 2 - 1
        return im
    
    @staticmethod
    def draw_pupil(im, lms, stroke=3):
        draw = im.copy()
        #draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
        pupil_center = np.zeros((2,))
        pnts_outerline = []
        pnts_innerline = []
        for i, lm in enumerate(np.squeeze(lms)):
            x, y = int(lm[0]), int(lm[1])

            if i < 8:
                draw = cv2.circle(draw, (y, x), stroke, (125,255,125), -1)
                pnts_outerline.append([y, x])
            elif i < 16:
                draw = cv2.circle(draw, (y, x), stroke, (125,125,255), -1)
                pnts_innerline.append([y, x])
                pupil_center += (y,x)
            elif i < 17:
                draw = cv2.drawMarker(draw, (y, x), (255,200,200), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=stroke, line_type=cv2.LINE_AA) # pupil center
            else:
                pass

        pupil_center = (pupil_center/8).astype(np.int32) # estimate pupil center
        draw = cv2.cv2.circle(draw, (pupil_center[0], pupil_center[1]), stroke, (255,255,0), -1)        
        draw = cv2.polylines(draw, [np.array(pnts_outerline).reshape(-1,1,2)], isClosed=True, color=(125,255,125), thickness=stroke//2)
        draw = cv2.polylines(draw, [np.array(pnts_innerline).reshape(-1,1,2)], isClosed=True, color=(125,125,255), thickness=stroke//2)
        return draw


        