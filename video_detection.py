import cv2
import numpy as np
from utils import *
from darknet import Darknet

# Set the location and name of the cfg file
cfg_file = '../data/cfg/crop_weed.cfg'

# Set the location and name of the pre-trained weights file
weight_file = '../data/weights/' + 'crop_weed_detection.weights'  # add weights file name here if you have your own

# Set the location and name of the object classes file
namesfile = '../data/names/obj.names'

# Load the network architecture
m = Darknet(cfg_file)

# Load the pre-trained weights
m.load_weights(weight_file)

# Load the COCO object classes
class_names = load_class_names(namesfile)

# Capture video from the webcam (you can replace 0 with a video file path)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Perform object detection on the frame
    boxes = detect_objects(m, frame, iou_thresh=0.4, nms_thresh=0.6)

    # Draw bounding boxes around detected objects, highlighting weeds
    for box in boxes:
        x, y, w, h = box[:4]
        cls_id = int(box[6])
        cls_name = class_names[cls_id]
        color = (0, 255, 0)  # Green color for highlighting weeds, adjust as needed

        if cls_name == "weed":
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
            cv2.putText(frame, cls_name, (int(x - w / 2), int(y - h / 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with bounding boxes
    cv2.imshow("Video", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
