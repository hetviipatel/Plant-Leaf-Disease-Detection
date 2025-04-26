import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

TF_ENABLE_ONEDNN_OPTS=0
# Load the trained model
model = load_model("trained_plant_disease_model.keras")

# Define class names (make sure this matches your model's class order)
class_names = ['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 
               'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 
               'Squash___Powdery_mildew', 'Potato___healthy', 
               'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 
               'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
               'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 
               'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
               'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 
               'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
               'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 
               'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 
               'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot', 
               'Corn_(maize)___healthy']

# Sort class names to ensure consistency
class_names.sort()

# Streamlit UI Setup
st.title("🌿 Plant Disease Prediction 🧑‍🔬")
st.write("Upload an image of a plant leaf and the model will predict the disease.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Progress Bar to show while prediction is being made
    st.write("Processing the image... Please wait.")
    progress_bar = st.progress(0)

    # Create a loop to simulate the process
    for i in range(100):
        time.sleep(0.05)  # Simulate some work being done (prediction process)
        progress_bar.progress(i + 1)

    # Image Preprocessing for model prediction
    image = Image.open(uploaded_file)
    image = image.resize((128, 128))  # Resize image to match input size for the model
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch format

    # Make prediction
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions, axis=1)[0]
    model_prediction = class_names[result_index]
    predicted_class_probability = np.max(predictions, axis=1)[0]

    # Show results with the improved UI
    st.markdown(f"## 🔍 Prediction Results")

    # Display the top prediction
    st.write(f"### Predicted Disease: **{model_prediction}**")
    st.write(f"**Confidence:** {predicted_class_probability * 100:.2f}%")

    # Show Top 3 predictions with confidence scores
    st.write("### Top 3 Predictions:")
    top_3_indices = np.argsort(predictions[0])[::-1][:3]
    for idx in top_3_indices:
        st.write(f"- **{class_names[idx]}**: {predictions[0][idx] * 100:.2f}%")

    # Additional information section (optional)
    st.write("---")
    st.markdown("### Additional Information 📝")
    st.write("""
    - **Tomato Late Blight** is a serious disease caused by the *Phytophthora infestans* pathogen.
    - **Pepper Bacterial Spot** is a bacterial disease affecting peppers.
    - For more information about the disease, refer to agricultural resources or consult a plant pathologist.
    """)

    # Display predictions using bar chart (to visualize confidence)
    st.write("### Prediction Confidence Bar Chart 📊")
    prediction_chart = np.array(predictions[0]) * 100
    st.bar_chart(prediction_chart, width=700, height=300, use_container_width=True)

# Styling and Branding
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f5f5f5;
        }
        .sidebar .sidebar-content {
            background-color: #A8DADC;
        }
        .stButton>button {
            background-color: #1D3557;
            color: white;
        }
        .stFileUploader {
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)
