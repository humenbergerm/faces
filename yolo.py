# thanks to https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/


# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

import utils


def detect_objects(args):

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.join(args.yolo, "coco.names")
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.join(args.yolo, "yolov3.weights")
	configPath = os.path.join(args.yolo, "yolov3.cfg")

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	files = utils.get_images_in_dir_rec(os.path.normpath(args.input))

	# Find all the faces and compute 128D face descriptors for each face.
	counter = 1
	total = len(files)
	for n, f in enumerate(files):
		print("Processing file ({}/{}): {}".format(counter, total, f))
		counter += 1

		# load our input image and grab its spatial dimensions
		image = cv2.imread(f)
		(H, W) = image.shape[:2]

		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# show timing information on YOLO
		print("[INFO] YOLO took {:.6f} seconds".format(end - start))

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > args.confidence:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, args.threshold)

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the image
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
					4, color, 6)

		# show the output image
		cv2.imshow("Image", utils.resizeCV(image, 800))
		cv2.waitKey(0)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', type=str, required=True, help="Input image directory.")
  parser.add_argument('--outdir', type=str, required=True, help="Output directory.")
  parser.add_argument('--yolo', type=str, help='Directory which contains the YOLO model.', required=True)
  parser.add_argument('--confidence', type=float, default=0.3, help='minimum probability to filter weak detections')
  parser.add_argument('--threshold', type=float, default=0.3, help='threshold when applying non-maxima suppression')
  args = parser.parse_args()

  if not os.path.isdir(args.input):
    print('args.input needs to be a valid folder containing images')
    exit()

  if not os.path.isdir(args.yolo):
    print('args.yolo needs to be a valid folder containing the YOLO model')
    exit()

  if not os.path.isdir(args.outdir):
    utils.mkdir_p(args.outdir)

  print('Detecting objects in {}'.format(args.input))
  detect_objects(args)
  print('Done.')

if __name__ == "__main__":
  main()