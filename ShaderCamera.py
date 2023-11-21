import cv2
import matplotlib.pyplot as plt # used to display filtered video
import numpy as np
from picamera2 import Picamera2, Preview
import skimage # used to add gaussian noise
import time # used for sleep funcionality

'''
Video Feed Filters
'''
# constants for nightvision filter
video_size = (480, 640)
empty_channel = np.zeros(video_size, dtype=np.uint8)
green_intensity = 0.57 # value found experimentally
noise_variance = 0.0025 # value found experimentally

# nightvision filter
def nightvision_fx(frame):
	# make the input image green and black
	g_channel = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	g_channel_reduced = (g_channel * green_intensity).astype(np.uint8)
	nightvision_frame = cv2.merge([empty_channel, g_channel_reduced, empty_channel])
	# add noise to the image to make it look grainy
	grainy_nightvision = skimage.util.random_noise(nightvision_frame, \
		mode='gaussian', var = noise_variance)
	return grainy_nightvision


'''
Video Input/Output Code
'''
# return the next frame (filtered)
def grab_frame():
	# get frame from the camera
	frame = picam.capture_array("main")
	# return filtered captured frame
	# TODO: allow user to change the applied effect in realtime
	return nightvision_fx(frame)
	
# Startup Camera
picam = Picamera2()
# Hide preview so that un-filtered video feed is not shown
picam.start(show_preview=False)
# give camera 3 seconds to startup
time.sleep(3)
	
# Display filtered video feed
# Note: below code is adapted from https://stackoverflow.com/questions/45025869/how-to-process-images-in-real-time-and-output-a-real-time-video-of-the-result
frame_plot = plt.imshow(grab_frame())
plt.ion()
while True:
	# display the latest filtered frame of the video
	frame_plot.set_data(grab_frame())
	# update display at roughly 30 fps
	plt.pause(0.03)
