import cv2
from matplotlib import pyplot as plt
import numpy as np

vid = cv2.VideoCapture(0)

fig, ax = plt.subplots()

while True:
	ret, frame = vid.read()
	frame = np.rot90(frame, 2)
	ax.clear()
	ax.imshow(frame)
	plt.pause(0.001)