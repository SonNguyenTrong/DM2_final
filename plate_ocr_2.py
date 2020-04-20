import cv2
import imutils
import numpy as np
import glob
import re
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt
from PIL import Image
import tempfile
import time
import argparse
import scipy.fftpack
cv_img = []
total_img = 0
right_img = 0
custom_config = r'--oem 3 --psm 6'


ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help='path to input image')
ap.add_argument('-c', '--config', default='yolov3.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default='yolov3.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='yolo.names',
                help='path to text file containing class names')
args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


for img in glob.glob("plate/*.jpg"):
    img_name = ''
    n= cv2.imread(img,cv2.IMREAD_COLOR)


    # plate detected by tesseract
    
    classes = None

    Width = n.shape[1]
    Height = n.shape[0]
    scale = 0.00392


    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(n, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(n, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        Cropped = n[round(y):round(y+h), round(x):round(x+w)]


    
    #Grayimage
    gray = cv2.cvtColor( Cropped, cv2.COLOR_BGR2GRAY)
    #Threshhold
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    plate_detected = pytesseract.image_to_string(Cropped, config=custom_config)
    plate_detected = re.sub('[^A-Za-z0-9]+', '', plate_detected)

    # name of the file
    img_name = re.sub('[^A-Za-z0-9]+', '', img)
    img_name = img_name.replace("plate", "")
    img_name = img_name.replace("jpg", "")

    total_img = total_img +1
    if(img_name == plate_detected):
        right_img = right_img + 1
        
   
    print('img real name: ', img_name, ', img detected name: ', plate_detected, ', total numb of imgs: ', total_img, ', total numb of right img: ', right_img)

print('percentage: ', (right_img/total_img)*100)