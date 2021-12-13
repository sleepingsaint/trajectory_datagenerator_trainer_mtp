import sys 
import os
import cv2
import argparse
import numpy as np
import pandas as pd
from halo import Halo
from utils import validatePath

def yoloDetect(frame, net, width, height, labels, colors, dataframe: pd.DataFrame, input_size=608, confidence_thres=0.5, nms_thres=0.5):
   
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > confidence_thres:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thres, nms_thres)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centerX = x + (w/2)
            centerY = y + (h/2)
            
            row = {"x": centerX / width, "y": centerY / height, "class_id": int(classIDs[i])}
             
            dataframe = dataframe.append(row, ignore_index=True)
             
            # color = [int(c) for c in colors[classIDs[i]]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame, dataframe


def generateData(weights, config, classes_file, input, input_size, conf_thres, nms_thres, frame_count, output_dir):
            
    validatePath(weights)
    validatePath(config)
    validatePath(classes_file)
    validatePath(input)
    
    # loading all the class names 
    classes = []
    with open(classes_file, "r") as f:
        classes = f.read().split("\n")
    
    # loading the darknet model
    yolo = cv2.dnn.readNetFromDarknet(config, weights)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8") 
    
    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        sys.exit("Error loading input file")
    
    # storing width and height of input video stream 
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    dataframe = pd.DataFrame(columns=["x", "y", "class_id"])
    curr_frame_count = 0
    
    spinner = Halo(text="Loading frames", spinner="dots")
    spinner.start() 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        curr_frame_count += 1
        spinner.text = f"Processing frame : {curr_frame_count}"
        # detecting bounding boxes
        frame, dataframe = yoloDetect(frame, yolo, frame_width, frame_height, classes, colors, dataframe, input_size=input_size, confidence_thres=conf_thres, nms_thres=nms_thres)
 
        # cv2.imshow("image", frame)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
        if frame_count is not None and frame_count == curr_frame_count:
            break
        
    spinner.stop()
    output = f"{int(frame_width)}_{int(frame_height)}_{input.split('.')[-2].split('/')[-1]}.csv"
    if output_dir is not None:
        output = os.path.join(output_dir, output)
    if os.path.exists(output) and os.path.isfile(output):
        os.remove(output)
    
    
    dataframe.to_csv(output, index=True)
    
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Generator for video file")
    parser.add_argument("--config", "-c", type=str, help="Path to config file")
    parser.add_argument("--classes", "-l", type=str, help="Path to classes file")
    parser.add_argument("--weights", "-w", type=str, help="Path to weights file")
    parser.add_argument("--input", "-i", type=str, help="Path to input video file")
    parser.add_argument("--size", '-s', type=int, default=608, help="Size of input of the trained model")
    parser.add_argument("--nms", type=float, default=0.5, help="NMS Threshold")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence Threshold")
    parser.add_argument("--frame_count", "-f", type=int, default=None, help="No. of frames to generate data")
    parser.add_argument("--output", '-o', type=str, default=None, help="Directory Path to save the dataset")
    
    args = parser.parse_args()
    generateData(args.weights, args.config, args.classes, args.input, args.size, args.conf, args.nms, args.frame_count, args.output)
