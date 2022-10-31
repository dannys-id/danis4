from imutils.video import FPS
import numpy as np
import argparse
import time
import cv2


net = cv2.dnn.readNet("yolov4-tiny-custom_best (2).weights", "yolov4-tiny-custom (1).cfg")

classes1 = []
with open("classes.txt", "r") as f:
    classes1 = f.read().splitlines()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt",
	help="path to Caffe 'deploy' prototxt file",default="MobileNetSSD_deploy.prototxt.txt")
ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
net1 = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] loading model...")
print("[INFO] starting video stream...")
cap = cv2.VideoCapture("rtsp://admin:RPBQIG@192.168.2.100:554/")
#ap = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()
green=(0,255,0)
kondisi=0
red=(0,0,255)
prev_frame_time = 0
new_frame_time = 0

while True:
	if (kondisi == 0):
		_, frame = cap.read()
		if frame is None:
			kondisi = 1
			pass
		else:
			new_frame_time = time.time()
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
				0.007843, (300, 300), 127.5)

			net1.setInput(blob)
			detections = net1.forward()
			a=0
			helm_safety=0
			sepatu_safety=0
			rompi_safety=0
			masker=0
			kacamata_safety=0
			sarungtangan_safety=0
			earmuff=0
			helm_sepeda=0
			frame1=frame.copy()
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]


			height, width, _ = frame1.shape
			blob1 = cv2.dnn.blobFromImage(frame1, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
			net.setInput(blob1)
			output_layers_names = net.getUnconnectedOutLayersNames()
			layerOutputs = net.forward(output_layers_names)

			boxes = []
			confidences = []
			class_ids = []

			for output in layerOutputs:
				for detection in output:
					scores = detection[5:]
					# print(scores)
					class_id = np.argmax(scores)
					confidence = scores[class_id]
					if confidence > 0.7:
						center_x = int(detection[0] * width)
						center_y = int(detection[1] * height)
						w = int(detection[2] * width)
						h = int(detection[3] * height)

						x = int(center_x - w / 2)
						y = int(center_y - h / 2)

						boxes.append([x, y, w, h])
						confidences.append((float(confidence)))
						class_ids.append(class_id)

			indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

			if len(indexes) > 0:
				for i in indexes.flatten():
					x, y, w, h = boxes[i]
					label = str(classes1[class_ids[i]])

					confidence = str(round(confidences[i], 2))
					if (label == "helm_safety"):
						color = (0, 0, 150)
						helm_safety=helm_safety+1
					elif (label == "sepatu_safety"):
						color = (0, 0, 255)
						sepatu_safety=sepatu_safety+1
					elif (label == "rompi_safety"):
						color = (0, 150, 0)
						rompi_safety=rompi_safety+1
					elif (label == "masker"):
						color = (0, 255, 0)
						masker=masker+1
					elif (label == "kacamata_safety"):
						color = (150, 0, 0)
						kacamata_safety=kacamata_safety+1
					elif (label == "sarungtangan_safety"):
						color = (255, 0, 0)
						sarungtangan_safety=sarungtangan_safety+1
					elif (label == "earmuff"):
						color = (150, 255, 0)
						earmuff=earmuff+1
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					cv2.putText(frame, label + " " , (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

			color = (0, 0xFF, 0xFF)
			thickness = 5
			cv2.putText(frame, "PendeteksiAPD ", (5, 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

			if(helm_safety==a):
				cv2.putText(frame, "Helm Safety :" + str(helm_safety), (5, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
			else:
				cv2.putText(frame, "Helm Safety :" + str(helm_safety), (5, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
			if(rompi_safety==a):
				cv2.putText(frame, "rompi_safety :" + str(rompi_safety), (5, 45),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
			else:
				cv2.putText(frame, "rompi_safety :" + str(rompi_safety), (5, 45),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
			if(sepatu_safety==(2*a)):
				cv2.putText(frame, "Sepatu Safety :" + str(sepatu_safety), (5, 60),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
			else:
				cv2.putText(frame, "Sepatu Safety :" + str(sepatu_safety), (5, 60),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
			if(masker==a):
				cv2.putText(frame, "Masker :" + str(masker), (180, 15),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
			else:
				cv2.putText(frame, "Masker :" + str(masker), (180, 15),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
			if(sarungtangan_safety==(2*a)):
				cv2.putText(frame, "sarungtangan_safety :" + str(sarungtangan_safety), (180, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
			else:
				cv2.putText(frame, "sarungtangan_safety :" + str(sarungtangan_safety), (180, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
			if(kacamata_safety==a):
				cv2.putText(frame, "Kacamata Safety :" + str(kacamata_safety), (180, 45),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
			else:
				cv2.putText(frame, "Kacamata Safety :" + str(kacamata_safety), (180, 45),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
			if(earmuff==(2*a)):
				cv2.putText(frame, "Earmuff :" + str(earmuff), (180, 60),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
			else:
				cv2.putText(frame, "Earmuff :" + str(earmuff), (180, 60),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break
			fps1 = 1 / (new_frame_time - prev_frame_time)
			prev_frame_time = new_frame_time
			print(fps1)
			fps.update()
	elif (kondisi == 1):
		cap = cv2.VideoCapture("rtsp://admin:RPBQIG@192.168.21.43:554/")
		_, img = cap.read()
		if img is None:
			print("tidak ada gambar")
		else:
			kondisi = 0
			pass

fps.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()