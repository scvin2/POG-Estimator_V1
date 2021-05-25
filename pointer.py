import math
from constants import *


# splits whole screen into several circle marks and returns the circle centers as list
def get_pts_list():

	circle_center_pts_list = []
	diameter = GAZE_SCREEN_WIDTH/NO_OF_CIRCLE_SCREEN_WIDTH
	circle_radius = math.floor(diameter/2)
	circle_diameter = circle_radius * 2
	no_of_circle_screen_height = math.floor(GAZE_SCREEN_HEIGHT/circle_diameter)

	x = 0
	y = 0

	# possible circle marks vertical wise
	while (y+circle_diameter) < GAZE_SCREEN_HEIGHT:
		
		x = 0
		
		# possible circle marks horizontal wise
		while (x+circle_diameter) < GAZE_SCREEN_WIDTH:
			
			center = [x+circle_radius, y+circle_radius]
			circle_center_pts_list.append(center)
			x += circle_diameter
		
		y += circle_diameter

	return circle_center_pts_list


# updates the position of the pointer on the screen
# updates are stabilized so the pointer moves only when it moves to a new circle mark
def update_cursor_pointer(radius, current_position, new_position):
	
	# check if new point is inside current point's circle or not
	# (x - center_x)^2 + (y - center_y)^2 < radius^2
	if ((new_position[0] - current_position[0])**2 + (new_position[1] - current_position[1])**2) < radius**2:
		return current_position
	return new_position
