import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model('saved_data/deepfake_model.h5')

# Streamlit UI
st.title("🧠 Deepfake Image Detector")
st.write("Upload a face image to check if it's Real or Fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to NumPy array
    img = np.array(image)

    # Preprocess
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]
    label = 'FAKE' if prediction >= 0.5 else 'REAL'
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    # Show result
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence * 100:.2f}%")



@st.cache_resource
def load_model_cached():
    return load_model('saved_data/deepfake_model.h5')

model = load_model_cached()
