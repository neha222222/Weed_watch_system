import streamlit as st
import cv2
import numpy as np
from utils import *
from darknet import Darknet

# Set the location and name of the cfg file
cfg_file = '../data/cfg/crop_weed.cfg'

# Set the location and name of the pre-trained weights file
weight_file = '../data/weights/crop_weed_detection.weights'  # add weights file name here if you have your own

# Set the location and name of the object classes file
namesfile = '../data/names/obj.names'

# Load the network architecture
m = Darknet(cfg_file)

# Load the pre-trained weights
m.load_weights(weight_file)

# Load the COCO object classes
class_names = load_class_names(namesfile)

# Streamlit app
st.title("Crop and Weed Detection App")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Convert uploaded image to OpenCV format
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to perform detection
    if st.button("Detect Weed"):
        # Detect objects in the image using your detection function
        boxes = detect_objects(m, image, iou_thresh=0.4, nms_thresh=0.6)

        # Check if any weed is detected
        weed_detected = any(obj[6] == class_names.index("weed") for obj in boxes)

        if weed_detected:
            st.error("Weed Detected!")
        else:
            st.success("No Weed Detected!")
            st.balloons()
