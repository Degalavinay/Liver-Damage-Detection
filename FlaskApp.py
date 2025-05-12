# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:15:46 2025

@author: degal
"""

# Import libraries
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, request, render_template_string
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model_path = 'C:/Users/degal/Desktop/Liver Damage Project/liver_damage_model.keras'
model = load_model(model_path)
print(f"Model loaded from {model_path}")

# Initialize Flask app
app = Flask(__name__)

# HTML template for the web interface
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Liver Disease Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        .container { max-width: 500px; margin: auto; }
        input[type="file"] { margin: 10px; }
        img { max-width: 300px; margin-top: 20px; }
        .result { font-size: 18px; color: #333; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Liver Disease Prediction</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <input type="submit" value="Predict">
        </form>
        {% if image_data %}
            <img src="data:image/jpeg;base64,{{ image_data }}" alt="Uploaded Image">
        {% endif %}
        {% if prediction %}
            <div class="result">Prediction: {{ prediction }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

# Preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction endpoint
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template_string(html_template, prediction="No image uploaded.")
        
        file = request.files['image']
        if file.filename == '':
            return render_template_string(html_template, prediction="No image selected.")
        
        if file:
            # Load and preprocess the image
            img = Image.open(file.stream).convert('RGB')
            processed_img = preprocess_image(img)
            
            # Make prediction
            prediction = model.predict(processed_img)
            class_idx = np.argmax(prediction, axis=1)[0]
            classes = ['Normal', 'CC', 'HCC']
            predicted_class = classes[class_idx]
            confidence = prediction[0][class_idx] * 100

            # Visualize prediction probabilities
            plt.figure(figsize=(6, 4))
            sns.barplot(x=prediction[0] * 100, y=classes, orient='h')
            plt.title('Prediction Probabilities')
            plt.xlabel('Confidence (%)')
            plt.ylabel('Class')
            plt.tight_layout()
            
            # Save the plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            prob_plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            # Convert image to base64 for display
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Update the HTML to include the probability plot
            extended_html = html_template.replace('</body>', f"""
                <div>
                    <img src="data:image/png;base64,{prob_plot_data}" alt="Probability Plot">
                </div>
                </body>
            """)
            
            return render_template_string(extended_html, 
                                         image_data=img_str, 
                                         prediction=f"{predicted_class} (Confidence: {confidence:.2f}%)")
    
    return render_template_string(html_template)

# Run the Flask app locally
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)