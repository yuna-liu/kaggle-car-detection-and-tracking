import streamlit as st
import cv2
import glob
import numpy as np

st.title("Car Detection YOLO Demo")

# Select dataset type
dataset_type = st.selectbox("Choose dataset", ["train", "val"])

# Define paths
dataset_path = f"./car_dataset/{dataset_type}/"
images_path = dataset_path + "images/"
labels_path = dataset_path + "labels/"

# List files
image_files = sorted(glob.glob(images_path + "*.jpg"))
label_files = sorted(glob.glob(labels_path + "*.txt"))

# Let user select image
img_choice = st.selectbox("Select an image", [f.split("/")[-1] for f in image_files])
img_index = [f.split("/")[-1] for f in image_files].index(img_choice)

# Load image
img_path = image_files[img_index]
lbl_path = label_files[img_index]
img = cv2.imread(img_path)
h, w, _ = img.shape

# Draw YOLO bounding boxes
with open(lbl_path, "r") as f:
    lines = f.readlines()

for line in lines:
    class_id, x_center, y_center, width, height = map(float, line.strip().split())
    x_center *= w
    y_center *= h
    width *= w
    height *= h
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

# Display image in Streamlit
st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")
