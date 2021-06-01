# https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/

# import the necessary packages
from helpers import *
from funcAP import *
from imutils import paths
import argparse
import imutils
import time
import dlib
import cv2
import os

printToTxt()
printToAP()
true_pos = 0
false_pos = 0
false_neg = 0
threshold = 0.5

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input image")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())

#na vypocet false_negative
for imagePath in paths.list_images(args["images"]):
	image = cv2.imread(imagePath)
	print(imagePath)
	# load dlib's HOG + Linear SVM face detector
	print("[INFO] loading HOG + Linear SVM face detector...")
	detector = dlib.get_frontal_face_detector()

	# load the input image from disk, resize it, and convert it from
	# BGR to RGB channel ordering (which is what dlib expects)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# perform face detection using dlib's face detector
	start = time.time()
	print("[INFO[ performing face detection with dlib...")
	rects = detector(rgb, args["upsample"])
	end = time.time()
	print("[INFO] face detection took {:.4f} seconds".format(end - start))

	# convert the resulting dlib rectangle objects to bounding boxes,
	# then ensure the bounding boxes are all within the bounds of the
	# input image
	boxes = [convert_and_trim_bb(image, r) for r in rects]

	# ak nemame hodnoty v boxes
	if (len(boxes) == 0):
		false_neg += 1
	# 	startX = 0
	# 	startY = 0
	# 	endX = 0
	# 	endY = 0
	# else:
	# 	# the predicted bounding boxes, red box
	# 	(startX, startY, w, h) = boxes[0]
	# 	endX = startX + w
	# 	endY = startY + h
	#
	# # hodnoty Ground-truth BB
	# photo_name = os.path.basename(imagePath)
	# line = readFromTxt(photo_name)
	# print(line)
	# x_r = int(line[1])
	# y_r = int(line[2])
	# w_r = int(line[3])
	# h_r = int(line[4])
	# x2_r = x_r + w_r
	# y2_r = y_r + h_r
	#
	# boxA = [startX, startY, endX, endY]
	# boxB = [x_r, y_r, x2_r, y2_r]
	# # compute the intersection over union and display it
	# iou = bb_intersection_over_union(boxA, boxB)
	#
	# # vypocet false negative
	# if (iou < threshold):
	# 	false_neg += 1

print("FALSE NEGATIVE ",false_neg)

for imagePath in paths.list_images(args["images"]):
	image = cv2.imread(imagePath)

	# load dlib's HOG + Linear SVM face detector
	print("[INFO] loading HOG + Linear SVM face detector...")
	detector = dlib.get_frontal_face_detector()

	# load the input image from disk, resize it, and convert it from
	# BGR to RGB channel ordering (which is what dlib expects)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# perform face detection using dlib's face detector
	start = time.time()
	print("[INFO[ performing face detection with dlib...")
	rects = detector(rgb, args["upsample"])
	end = time.time()
	print("[INFO] face detection took {:.4f} seconds".format(end - start))

	# convert the resulting dlib rectangle objects to bounding boxes,
	# then ensure the bounding boxes are all within the bounds of the
	# input image
	boxes = [convert_and_trim_bb(image, r) for r in rects]

	# ak nemame hodnoty v boxes
	if (len(boxes) == 0):
		# startX = 0
		# startY = 0
		# endX = 0
		# endY = 0
		continue
	else:
		# the predicted bounding boxes
		(startX, startY, w, h) = boxes[0]
		endX = startX + w
		endY = startY + h

	# red box
	# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

	# hodnoty Ground-truth BB
	photo_name = os.path.basename(imagePath)
	line = readFromTxt(photo_name)
	print(line)
	x_r = int(line[1])
	y_r = int(line[2])
	w_r = int(line[3])
	h_r = int(line[4])
	x2_r = x_r + w_r
	y2_r = y_r + h_r
	print(photo_name)
	# TODO read rectangle from file Ground-truth BB, green box
	# cv2.rectangle(image, (x_r, y_r), (x2_r, y2_r), (0, 255, 0), 2)

	boxA = [startX, startY, endX, endY]
	boxB = [x_r, y_r, x2_r, y2_r]
	# compute the intersection over union and display it
	iou = bb_intersection_over_union(boxA, boxB)
	# cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	# vypocet Precision a Recall
	if (iou >= threshold):
		true_pos += 1
	elif (iou < threshold):
		false_pos += 1

	calcPrecisionRecall(photo_name, true_pos, false_pos, false_neg, 1)

	# show the output image
	width = endX - startX
	height = endY - startY

	# vyprintovanie do konzoly
	print(photo_name)
	print("Prediction BB ", startX, startY, endX, endY)
	print("Ground truth BB ", x_r, y_r, x2_r, y2_r)

	# vyprintovanie do priecinku image.txt,moje udaje (z prediction bb)
	appendToTxt(photo_name, startX, startY, width, height)

	# # show the output image
	# cv2.imshow("Output", image)
	# cv2.waitKey(0)

print("FALSE NEGATIVEs ",false_neg)
print("TRUE POSITIVES ",true_pos)
print("FALSE POSITIVES ",false_pos)
plot_model(precisionArray,recallArray)

# python mainDlib.py -i C:\Users\Lenovo\PycharmProjects\secondProjectFaceDetection\foto\003877.jpg

# python mainDlib.py -i C:\Users\Lenovo\PycharmProjects\secondProjectFaceDetection\foto

# python mainDlib.py -i C:\Users\Lenovo\Desktop\bakalarka\Celeb