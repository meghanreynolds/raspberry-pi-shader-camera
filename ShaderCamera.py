import cv2
import datetime # used for vhs tape filter timestamp
import matplotlib.pyplot as plt # used to display filtered video
import numpy as np
from picamera2 import Picamera2, Preview
from PIL import Image, ImageDraw, ImageFont # used for vhs tape filter
import skimage # used to add gaussian noise
import sys # used to read in desired filter from command line
import time # used for sleep funcionality

'''
Video Feed Filters
'''
video_size = (480, 640)
white = np.full(3, 255)

### CARTOON FILTER###
def cartoon_filter(frame):
	frame = frame[:,:,0:3]
	# Convert to grayscale and apply median blur to reduce image noise
	grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	grayimg = cv2.medianBlur(grayimg, 5)
	# Get the edges
	edges = cv2.adaptiveThreshold(grayimg.astype(np.uint8), 255, \
		cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
	# Convert to a cartoon version
	color = cv2.bilateralFilter(frame.astype(np.uint8), 9, 250, 250)
	cartoon = cv2.bitwise_and(color, color, mask=edges)
	return cartoon

### VHS TAPE FILTER ###
# Constants for vhs tape filter
font_size = 20
font_absolute_path = "/home/mreynold/raspberry-pi-shader-camera/fonts/VCR_OSD_MONO.ttf"
font = ImageFont.truetype(font_absolute_path, 20)
timestamp_x = 10
timestamp_y = video_size[0] - font_size - 10
timestamp_locaiton = (timestamp_x, timestamp_y)
# Add recording label
recording_label_text = "REC"
recording_dot_x = 10
recording_dot_y = 10

# Vhs tape filter
def vhs_filter(frame):
	# remove alpha channel
	frame = frame[:,:,0:3] 
	# convert image to np array (TODO see if needed now)
	np_image = np.array(frame)
	
	# Add scanlines: Skip every other line
	np_image[::2,:,:] = np_image[::2, :, :] * 0.3
	
	# Adjust color balance: Slightly increase red, decrease blue
	np_image[:,:,0] = np_image[:,:,0] * 1.1 # Red Channel
	np_image[:,:,2] = np_image[:, :, 2] * 0.9 # Blue Channel
	
	# Adding noise
	noise = np.random.normal(0, 25, np_image.shape)
	np_image = np.clip(np_image + noise, 0, 255)
	
	# Converting back to PIL Image
	vhs_image = Image.fromarray(np_image.astype('uint8'), 'RGB')
	
	# Adding timestamp
	draw = ImageDraw.Draw(vhs_image)

	# Formatting the timestamp text
	text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	
	# Positioning the text at the bottom left corner 
	# TODO move x and y out of the function
	draw.text(timestamp_locaiton, text, (255, 255, 255), font=font)
	
	# Add recording label
	# Drawing a red dot
	draw.ellipse([(recording_dot_x, recording_dot_y), \
		(recording_dot_x + 10, recording_dot_y + 10)], fill=(255, 0, 0))
	# Drawing the label text next to the red dot
	draw.text((recording_dot_x + 15, recording_dot_y - 5), \
		recording_label_text, (255, 255, 255), font=font)
	
	return vhs_image
	

### NIGHTVISION FILTER ###
# constants for nightvision filter
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

### BLUEPRINT FILTER ###
# constants for blueprint filter
grid_size = 11 # found by experimentaiton
grid_strength = 0.4 # found by experimentation
blueprint_blue = (48, 48, 225) # found by experimentation
edge_threshold = 0.999 # found by experimentation

# create the blueprint grid
attenuated_white = grid_strength * white
blueprint_grid = np.full((video_size[0], video_size[1], 3), \
	blueprint_blue)
for i in range(0, blueprint_grid.shape[0]):
	for j in range(0, blueprint_grid.shape[1]):
		if (i % grid_size == 0 or j % grid_size == 0):
			blueprint_grid[i][j] = attenuated_white

# create sobel filter
sobel_filter = np.zeros((3,3))
sobel_filter[0][0] = -1
sobel_filter[1][0] = -2
sobel_filter[2][0] = -1
sobel_filter[0][2] = 1
sobel_filter[1][2] = 2
sobel_filter[2][1] = 1
sobel_filter *= (1/8)

# blueprint filter
def blueprint_fx(frame):
	frame = frame[:,:,0:3]
	# apply smoothing so we only get stronger edges
	blurred = cv2.GaussianBlur(frame, (13, 13), -1)
	# apply sobel filter to intensity image to get edges
	intensity_img = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
	blueprint_outline = cv2.filter2D(intensity_img.astype('float'), -1, \
		sobel_filter)
	# threshold to only draw strong edges, edges drawn as white pixels
	blueprint_outline = np.abs(blueprint_outline)
	blueprint_outline[blueprint_outline > edge_threshold] = 255
	blueprint_outline[blueprint_outline != 255] = 0
	# make outline into rgb data
	mask = np.zeros(frame.shape)
	mask[:,:,0] = blueprint_outline
	mask[:,:,1] = blueprint_outline
	mask[:,:,2] = blueprint_outline
	# add edges ontop of blueprint grid
	bp_img = blueprint_grid + mask.astype('int')
	bp_img = np.clip(bp_img, 0, 255)
	return bp_img

### SKETCHBOOK FILTER ###
def sketchbook_filter(frame):
	# convert to grayscale
	grayscale = np.array(np.dot(frame[..., :3], [0.299, 0.587, 0.114]), dtype = np.uint8)
	grayscale = np.stack((grayscale,) * 3, axis = -1)

	# invert the image
	invert_img = 255 - grayscale

	# blur the image
	blur_img = cv2.GaussianBlur(invert_img, ksize = (0, 0), sigmaX = 5)

	# create the sketchbook filter
	result = grayscale * 255.0 / (255.0 - blur_img)
	result[result > 255] = 255
	result[grayscale == 255] = 255
	result_img = result.astype('uint8')

	# sharpen the image
	kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    
	sharpenImage = cv2.filter2D(result_img, -1, kernel)

	return sharpenImage

### EMBOSSED FILTER ###
def embossed_filter(frame):
	kernel = np.array([[0, -3, -3],
                      [3, 0, -3],
                       [3, 3, 0]]) 

	emboss_img = cv2.filter2D(frame, -1, kernel = kernel)

	return emboss_img

### 8-BIT FILTER ###
def bit_filter(frame):
	height, width = frame.shape[:2]

	w, h = (256, 256)

	temp = cv2.resize(frame, (w, h), interpolation = cv2.INTER_LINEAR)

	output = cv2.resize(temp, (width, height), interpolation = cv2.INTER_NEAREST)

	return output

'''
Video Input/Output Code
'''
index_of_filter = 1
cartoon_filter_code = "toon"
nightvision_filter_code = "night"
vhs_filter_code = "vhs"
blueprint_filter_code = "bp"
sketchbook_filter_code = "sketch"
embossed_filter_code = "emboss"
bit_filter_code = "bit"

# set the video filter
def get_filter(filter_code):
	if (filter_code == cartoon_filter_code):
		return cartoon_filter
	elif (filter_code == nightvision_filter_code):
		return nightvision_fx
	elif (filter_code == blueprint_filter_code):
		return blueprint_fx
	elif (filter_code == vhs_filter_code):
		return vhs_filter
	elif (filter_code == sketchbook_filter_code):
		return sketchbook_filter
	elif (filter_code == embossed_filter_code):
		return embossed_filter
	elif (filter_code == bit_filter_code):
		return bit_filter
	else:
		return nil

index_of_filter = 1
video_filter = get_filter(sys.argv[index_of_filter])
# return the next frame (filtered)
def grab_frame():
	# get frame from the camera
	frame = picam.capture_array("main")
	# return filtered captured frame
	return video_filter(frame)
	
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
