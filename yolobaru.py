import subprocess

import cv2
import numpy as np
from playsound import playsound
from datetime import datetime
import time
import imutils
net = cv2.dnn.readNet("yolov4-tiny-custom_best (2).weights", "yolov4-tiny-custom (1).cfg")

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
a=0
cap = cv2.VideoCapture("rtsp://admin:RPBQIG@192.168.2.100:554/")
#cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
prev_frame_time = 0

new_frame_time = 0
kondisi=0
nomask=False
while True:
    if(kondisi==0):
        _, img = cap.read()
        if img is None:
            # print("tidak ada gambar")
            kondisi=1
            pass
        else:
            height, width, _ = img.shape
            new_frame_time = time.time()
            #print("oke")

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
                    #print(scores)
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.9:
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
                    cv2.putText(img, label + " " , (x, y + 20), font, 2, (255, 255, 255), 2)

            color = (0, 0xFF, 0xFF)
            thickness = 5
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            print(fps)

            cv2.imshow('Frame', img)
            key = cv2.waitKey(1)
            if key==27:
                break
    elif(kondisi==1):
        cap = cv2.VideoCapture("rtsp://admin:RPBQIG@192.168.1.100:554/")
        _, img = cap.read()
        if img is None:
            print("tidak ada gambar")
        else:
            kondisi=0
            pass

cap.release()
cv2.destroyAllWindows()