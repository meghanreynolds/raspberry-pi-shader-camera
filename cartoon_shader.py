#Method 1 with OpenCV
import cv2
def cartoon_filter1(frame):
    #Convert to grayscale and apply median blur to reduce image noise
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.medianBlur(frame, 5)
    #Get the edges 
    edges = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
    #Convert to a cartoon version
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

#Method 2 with scikit-image
from skimage import io, color
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb

def cartoon_filter2(frame):
    # Convert to grayscale
    grayimg = color.rgb2gray(frame)
    # Apply the felzenszwalb segmentation
    segments_fz = felzenszwalb(frame, scale=100, sigma=0.5, min_size=50)
    # Convert to a cartoon version
    cartoon = color.label2rgb(segments_fz, frame, kind='avg')
    return cartoon

#Method 3 with Pillow
from PIL import Image, ImageOps, ImageFilter
def cartoon_filter3(frame):
    # Convert to a cartoon version
    cartoon = ImageOps.posterize(frame, 2)
    return cartoon