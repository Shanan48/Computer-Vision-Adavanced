# import the necessary packages
import cv2

"""In order to build face recognition software, Jeremy has to
use the built-in Haar cascade classifiers in OpenCV. Luckily for him, these classifiers have already been pre-trained
to recognize faces!"""

class FaceDetector:
	def __init__(self, faceCascadePath):
		# load the face detector
		self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

	def detect(self, image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30)):
		# detect faces in the image
		rects = self.faceCascade.detectMultiScale(image,
			scaleFactor = scaleFactor, minNeighbors = minNeighbors,
			minSize = minSize, flags = cv2.CASCADE_SCALE_IMAGE)

		# return the rectangles representing bounding
		# boxes around the faces
		return rects

"""scaleFactor: How much the image size is reduced at
each image scale. This value is used to create the scale
pyramid in order to detect faces at multiple scales
in the image (some faces may be closer to the foreground, and thus be larger; other faces may be smaller
and in the background, thus the usage of varying
scales). A value of 1.05 indicates that Jeremy is reducing the size of the image by 5% at each level in the
pyramid.

minNeighbors: How many neighbors each window
should have for the area in the window to be considered a face. The cascade classifier will detect multiple
windows around a face. This parameter controls how
many rectangles (neighbors) need to be detected for
the window to be labeled a face.

minSize: A tuple of width and height (in pixels) indicating the minimum size of the window. Bounding
boxes smaller than this size are ignored. It is a good
idea to start with (30, 30) and fine-tune from there.

Detecting the actual faces in the image is handled on
Line 8 by making a call to the detectMultiScale method of
Jeremy’s classifier created in the constructor of the FaceDetector class. He supplies his scaleFactor, minNeighbors,
and minSize, then the method takes care of the entire face
detection process for him!
"""