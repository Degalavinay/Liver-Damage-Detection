import os
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Configuration
IMG_SIZE = (224, 224)
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'VGG19_final_model.keras')
CLASSES = ['Normal', 'CC', 'HCC']  # Class labels

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    st.error(f"Error loading model from {MODEL_PATH}: {e}")
    st.stop()

# Function to preprocess a single image
def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to convert image to base64 for HTML display
def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Streamlit app
# Custom CSS for styling
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
    font-family: 'Arial', sans-serif;
}
.header {
    background-color: #4CAF50;
    color: white;
    padding: 20px;
    text-align: center;
    border-radius: 10px;
    margin-bottom: 20px;
}
.container {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin: 10px 0;
}
.upload-btn {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    cursor: pointer;
    margin-top: 10px;
}
.upload-btn:hover {
    background-color: #45a049;
}
.result-box {
    background-color: #e8f5e9;
    padding: 15px;
    border-radius: 5px;
    margin-top: 20px;
}
.probability {
    font-size: 16px;
    margin: 5px 0;
}
.predicted-class {
    font-size: 18px;
    font-weight: bold;
    color: #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# Header using HTML
st.markdown("""
<div class="header">
    <h1>Liver Damage Detection Model</h1>
    <p>Upload a histopathology image to classify it as Normal, CC, or HCC</p>
</div>
""", unsafe_allow_html=True)

# Main container for the app
st.markdown('<div class="container">', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.markdown('<h3>Uploaded Image</h3>', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)  # Reverted to use_column_width

    # Preprocess and predict
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_probabilities = prediction[0]

    # Display prediction results with styling
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown('<h3>Prediction Results</h3>', unsafe_allow_html=True)
    for cls, prob in zip(CLASSES, class_probabilities):
        st.markdown(f'<p class="probability">{cls}: {prob:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="predicted-class">Predicted Class: {CLASSES[predicted_class]} (Confidence: {class_probabilities[predicted_class]:.4f})</p>', unsafe_allow_html=True)
    
    # Visualize the image with prediction
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Predicted: {CLASSES[predicted_class]} ({class_probabilities[predicted_class]:.4f})")
    ax.axis('off')
    st.pyplot(fig)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Close container
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="header" style="background-color: #333; margin-top: 20px;">
    <p style="color: white;">Developed using Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)