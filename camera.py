import pyk4a
from pyk4a import Config, PyK4A


# set the RGBD camera for testing
# configuration azure kinect
def configure_camera():
	k4a = PyK4A( 
				Config(
						color_resolution=pyk4a.ColorResolution.RES_720P,
						color_format=pyk4a.ImageFormat.COLOR_BGRA32,
						camera_fps=pyk4a.FPS.FPS_15,
						depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
						synchronized_images_only=True,
				))
	
	return k4a