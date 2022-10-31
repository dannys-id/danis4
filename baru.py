import cv2
import queue
import time
import threading
import numpy as np
q = queue.Queue()
net = cv2.dnn.readNet("yolov4-tiny-custom_best (2).weights", "yolov4-tiny-custom (1).cfg")

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
a=0
font = cv2.FONT_HERSHEY_PLAIN

def Receive():
    print("start Reveive")
    #cap = cv2.VideoCapture("rtsp://admin:RPBQIG@192.168.100.199:554/ch0.0h264")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def Display():
    print("Start Displaying")
    while True:
        if q.empty() != True:
            frame = q.get()
            height, width, _ = g=frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
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
                    if confidence > 0.2:
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
                    label = str(classes[class_ids[i]])

                    confidence = str(round(confidences[i], 2))
                    if (label == "helm_safety"):
                        color = (0, 0, 150)
                    elif (label == "sepatu_safety"):
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
                        color = (150, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

            color = (0, 0xFF, 0xFF)
            thickness = 5
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()