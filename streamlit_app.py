import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page config
st.set_page_config(
    page_title="Tomato Plant Disease Classifier",
    page_icon="🍅",
    layout="centered"
)

# Title and description
st.title("🍅 Tomato Plant Disease Classification")
st.write("Upload an image of a tomato leaf to detect diseases")

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tomato5.h5", compile=False)

MODEL = load_model()

CLASS_NAMES = [
    'Tomato Early Blight', 
    'Tomato Late Blight', 
    'Tomato Leaf Mold', 
    'Tomato Yellow Leaf Curl Virus',
    'Tomato Healthy'
]

# File uploader
uploaded_file = st.file_uploader(
    "Choose a tomato leaf image.. .", 
    type=['jpg', 'png', 'jpeg']
)

if uploaded_file is not None:
    # Display image
    col1, col2 = st. columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    with st.spinner('Analyzing image...'):
        # Preprocess
        img_resized = image.resize((256, 256))
        img_array = np.array(img_resized)
        img_batch = np.expand_dims(img_array, 0)
        
        # Predict
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np. max(predictions[0])
    
    # Display results
    with col2:
        st.subheader("Results")
        st.metric("Prediction", predicted_class)
        st.metric("Confidence", f"{confidence:.2%}")
        
        # Show all predictions
        st.subheader("All Predictions")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {predictions[0][i]:.2%}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • TensorFlow • Python")
