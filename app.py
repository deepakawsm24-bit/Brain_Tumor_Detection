import streamlit as st
import tensorflow as tf
import numpy as np
form PIL import Image

# Page Title
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
st.title("🧠 Brain Tumor Detection")

# Load Trained Model
model = tf.keras.models.load_model("brain_tumor_model.keras")
st.write("Upload an MRI Brain Image to Detect Tumor")

# Upload MRI Image
uploaded_file = st.file_uploaded("Upload Brain MRI Image",type = ["jpg","jpeg","png"])

# Class Names
class_names = ["glioma_tumor", "meningioma_tumor",  "no_tumor", "pituitary_tumor"]

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image,caption = "Uploaded MRI Image", use_container_width=True)
  img = image.resize(224,224))
  img = np.array(img)/255.0
  img = np.expand_dims(img,axis=0)
  prediction = model.predict(img)
  predicted_class = class_names[np.argmax(prediction)]
  confidence = np.max(prediction)*100
  st.success(f"Prediction:{predicted_class}")
  st. write(f"Confidence:{condfidence:.2f}%")









