
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load model
model = load_model("models/final_defect_model.h5")

st.title("AI Defect Detection (Image Upload)")
uploaded_file = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.resize(image, (224, 224))
    img = img_to_array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    label = "Defective" if pred > 0.5 else "Non-defective"
    confidence = pred if pred > 0.5 else 1 - pred

    st.image(image, caption=f"Prediction: {label} ({confidence*100:.2f}%)", channels="BGR")
