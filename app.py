import streamlit as st
import cv2
import glob
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from ultralytics import YOLO

# ---------------------------
# Title
# ---------------------------
st.title("Car Detection & ML Demo")

# ---------------------------
# Dataset selection
# ---------------------------
dataset_type = st.selectbox("Choose dataset", ["train", "val"])
dataset_path = f"./car_dataset/{dataset_type}/"
images_path = dataset_path + "images/"
labels_path = dataset_path + "labels/"

# ---------------------------
# Load image files
# ---------------------------
image_files = sorted(glob.glob(images_path + "*.jpg"))
label_files = sorted(glob.glob(labels_path + "*.txt"))

# ---------------------------
# Sidebar: Next / Previous image
# ---------------------------
if 'img_index' not in st.session_state:
    st.session_state.img_index = 0

def prev_image():
    if st.session_state.img_index > 0:
        st.session_state.img_index -= 1

def next_image():
    if st.session_state.img_index < len(image_files) - 1:
        st.session_state.img_index += 1

st.sidebar.button("Previous", on_click=prev_image)
st.sidebar.button("Next", on_click=next_image)

img_index = st.session_state.img_index
img_path = image_files[img_index]
lbl_path = label_files[img_index]

st.sidebar.write(f"Image: {img_path.split('/')[-1]} ({img_index+1}/{len(image_files)})")

# ---------------------------
# Load image
# ---------------------------
img = cv2.imread(img_path)
h, w, _ = img.shape

# ---------------------------
# Draw ground truth boxes (Green)
# ---------------------------
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
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)  # Green = GT

# ---------------------------
# YOLOv8 prediction (Blue)
# ---------------------------
st.sidebar.write("Running YOLOv8 detection...")
model_yolo = YOLO("yolov8n.pt")  # pretrained model
results = model_yolo(img_path)

for box, score in zip(results[0].boxes.xyxy, results[0].boxes.conf):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)  # Blue = predicted
    cv2.putText(img, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

# ---------------------------
# Pretrained classifier (ResNet18)
# ---------------------------
st.sidebar.write("Running classifier (ResNet18)...")
model_cls = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_cls.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img_pil = Image.open(img_path).convert("RGB")
input_tensor = transform(img_pil).unsqueeze(0)

with torch.no_grad():
    outputs = model_cls(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    conf, predicted = torch.max(probs, 1)

st.sidebar.write(f"Classifier Prediction: ID {predicted.item()} with confidence {conf.item():.2f}")

# ---------------------------
# Legend
# ---------------------------
st.sidebar.markdown("### Legend")
st.sidebar.markdown("🟩 Ground Truth Box")
st.sidebar.markdown("🟦 YOLO Predicted Box with confidence")

# ---------------------------
# Display final image
# ---------------------------
st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")
