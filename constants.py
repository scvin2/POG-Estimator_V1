from pathlib import Path


FILE_PATH = Path(__file__).parent.resolve()  # current file path

# resolution values
CAPTURE_IMAGE_WIDTH = 1280  # width of the images captured by the camera
CAPTURE_IMAGE_HEIGHT = 720  # height of the images captured by the camera
MAX_CAPTURE_DISTANCE = 1000  # set maximum operating distance
GAZE_SCREEN_WIDTH = 1920  # resolution of the gaze screen
GAZE_SCREEN_HEIGHT = 1080  # resolution of the gaze screen

VIDEO_OUTPUT_FRAME_WIDTH = 1920  # test video output resolution
VIDEO_OUTPUT_FRAME_HEIGHT = 1080  # test video output resolution

NO_OF_CIRCLE_SCREEN_WIDTH = 25  # no of circle marks the screen is divided into

CAMERA_FOCAL_X = 600  # camera parameter
CAMERA_FOCAL_Y = 600  # camera parameter

MIN_DEPTH_ERROR_DISTANCE = 200  # minimum operating distance

FACE_DEPTH_IMG_CROP_SHAPE = (112, 112)  # shape of the face depth image given to the neural network as input
FACE_DATA_SHAPE = (112, 112, 3)  # shape of the face data given to the neural network as input
EYE_DEPTH_IMG_CROP_SHAPE = (112, 112)  # shape of the eye depth image given to the neural network as input
EYE_DATA_SHAPE = (112, 112, 3)  # shape of the eye data given to the neural network as input

NO_OF_FRAMES_PER_MARK = 20  # no of frames recorded for every mark - used when capturing data for training
 
DATA_DIR = FILE_PATH / "data"  # captured data save path
EXTRACT_DIR = FILE_PATH / "extracted_data"  # save path for extracted information from the captured data

TRAIN_PARTICPANT_LIST = ["p3", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12"]  # participant data used for training
VALIDATE_PARTCIPANT_LIST = ["p1", "p2", "p4", "p13"]  # participant data used for validation
TEST_PARTICIPANT_LIST = []  # participant data used for testing
# TIME_FRAMES_PER_INPUT = 3

SCALING = "m1p1"  # scaling format used for input data, can be "minmax" or "m1p1"

POINTER_RADIUS = 60  # test pointer radius

POINTER_SHOW_RADIUS = POINTER_RADIUS
MAX_CAPTURE_IMAGE_WIDTH_VAL = CAPTURE_IMAGE_WIDTH - 1
MAX_CAPTURE_IMAGE_HEIGHT_VAL = CAPTURE_IMAGE_HEIGHT - 1
MAX_GAZE_SCREEN_WIDTH_VAL = GAZE_SCREEN_WIDTH-1
MAX_GAZE_SCREEN_HEIGHT_VAL = GAZE_SCREEN_HEIGHT-1
