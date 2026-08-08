import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠"
)

# Title
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI Brain Image to Detect Tumor")

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Upload MRI Image
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

# Class Names
class_names = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded MRI Image",
        width="stretch"
    )

    # Resize image
    img = image.resize((224, 224))

    # Convert image to numpy array
    img = np.array(img) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    # Get predicted class
    predicted_class = class_names[np.argmax(prediction)]

    # Get confidence
    confidence = float(np.max(prediction)) * 100

    # Display result
    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")









