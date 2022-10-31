import subprocess

import cv2
import numpy as np
import threading
import time
#q=queue.Queue()
from playsound import playsound
from datetime import datetime
net = cv2.dnn.readNet("yolov4-tiny-custom_best (2).weights", "yolov4-tiny-custom (1).cfg")

classes = []
global now,prev
now=0
prev=0
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
a=0
cap = cv2.VideoCapture("rtsp://admin:RPBQIG@192.168.1.100:554/ch0.0h264")
#cap = cv2.VideoCapture(0)

#cap = cv2.imread('helm.jpg')

font = cv2.FONT_HERSHEY_PLAIN

nomask=False
while True:
    _, img = cap.read()
    height, width, _ = img.shape
    print("oke")

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            print(scores)
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            confidence = str(round(confidences[i], 2))
            if(label=="helm_safety"):
                color = (0,0,150)
            elif(label=="sepatu_safety"):
                color = (0, 0, 255)
            elif (label == "rompi"):
                color = (0, 150, 0)
            elif (label == "masker"):
                color = (0, 255, 0)
            elif (label == "kacamata_safety"):
                color = (150, 0, 0)
            elif (label == "sarung_tangan"):
                color = (255, 0, 0)
            else:
                color=(150,255,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    color = (0, 0xFF, 0xFF)
    thickness = 5

    cv2.imshow('Frame', img)
    key = cv2.waitKey(1)
    if key==27:
        break



cap.release()
cv2.destroyAllWindows()