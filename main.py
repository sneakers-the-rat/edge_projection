#!/usr/bin/env python

"""
Code to grab frames from a webcam, reduce them to edges, 
colorize the edges, and fade that color with the age of the edges

Intended for projection in the LISB atrium at the UO.
"""

__version__ = 1.0
__author__ = "Jonny Saunders"

import cv2
import numpy as np
from skimage import img_as_float
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import imops
from itertools import count

##############
# parameters

n_traces     = 10  # number of colored traces to keep in each image
n_skipframes = 2   # number of frames to skip between each trace
# need to uncomment lines below to skip frames
canny_sig    = 4   # sigma to use w/ gaussian blur in edge detectoin
canny_high   = 0.3 # canny high threshold (8 bit)
canny_low    = 0.1 # canny low threshold


#################
# Colormap

# cdict = {'red':   ((0., 0.0416,0.0416),
#                    (0.365079, 1.000000, 1.000000),
#                    (1.0,  1.,  1.)),
#          'green': ((0., 0.000000, 0.000000),
#                    (0.365079, 0.000000, 0.000000),
#                    (0.746032, 1.000000, 1.000000),
#                    (1.0, 1.000000, 1.000000)),
#          'blue':  ((0., 0., 0.),
#                    (0.746032, 0.000000, 0.000000),
#                    (1.0, 1.0, 1.0))}
# custom_cmap = LinearSegmentedColormap('custom_cmap',cdict)

#yarite "custom"
custom_cmap = cm.hot

##############
# cmap values - decrement values by this each time to fade old traces
color_decr = 1./n_traces

# open video and get one frame to make preallocated arrays
vid = cv2.VideoCapture(0)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# preallocate a frame to store edge values
display_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

# open opencv window & fullscreen
cv2.namedWindow('main', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('main', cv2.WND_PROP_FULLSCREEN,
cv2.WINDOW_FULLSCREEN)

# counter for skipping frames if we wanted to
nframes = count()

while True:
    ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # uncomment below if you want to skip frames between drawing edges
    # current_frame = nframes.next()
    # if current_frame % n_skipframes != 0:
    #     continue
    # if current_frame > 1000000:
    #     nframes = count()
    
    # detect edges w/ a modified skimage canny...
    # in this case just uses opencv for convolutions 
    # & uses Scharr kernel instead of Sobel
    edges = imops.scharr_canny(img_as_float(frame), sigma=canny_sig, 
                                        high_threshold = canny_high,
                                        low_threshold  = canny_low)
    
    # decay old traces, then set new trace to 1.
    display_frame = display_frame-color_decr
    display_frame = np.clip(display_frame, 0.,1.)
    display_frame[edges] = 1.0

    # convert 1d edge array to RGB using the above colormap
    # filter 0:3 in 2nd dim because cmaps return alpha channel
    # flip along the 2nd dimension b/c opencv is BGR for some reason
    cv2.imshow('main', np.flip(custom_cmap(display_frame)[:,:,0:3], 2))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

