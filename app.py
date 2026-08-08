import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(page_title="Brain Tumor Detection AI", page_icon="🧠", layout="wide")

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""<style>.main-title { font-size: 42px; font-weight: 700; margin-bottom: 5px;}

.subtitle {  font-size: 18px;   color: #666; margin-bottom: 25px;}

.section-title {  font-size: 25px;  font-weight: 650;  margin-top: 20px;   margin-bottom: 12px;}

.result-card {   padding: 22px;    border-radius: 12px;   border: 1px solid #d9e2ec;   background-color: #f8fbff;   margin-top: 10px;}

.info-card {    padding: 18px;    border-radius: 12px;    border: 1px solid #e1e7ef;    background-color: #ffffff;}

.small-text {    color: #666;    font-size: 14px;}
 
.disclaimer {   padding: 15px;    border-radius: 10px;    background-color: #fff8e6;    border: 1px solid #f0d98c;    font-size: 14px;} </style>""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown( '<div class="main-title">🧠 Brain Tumor Detection AI</div>',    unsafe_allow_html=True)

st.markdown( '<div class="subtitle">' 'Deep Learning Based Brain MRI Image Classification System' '</div>',    unsafe_allow_html=True)

st.markdown("---")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.write("**Model:** Deep Learning")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Input Size:** 224 × 224")
    st.write("**Number of Classes:** 4")

    st.markdown("---")

    st.subheader("🧠 Classification Classes")

    st.write("• Glioma Tumor")
    st.write("• Meningioma Tumor")
    st.write("• Pituitary Tumor")
    st.write("• No Tumor")

    st.markdown("---")

    st.success("🟢 Model Loaded")
    st.info("Ready for MRI Analysis")


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(  '<div class="section-title">📤 Upload MRI Scan</div>',    unsafe_allow_html=True)

st.write( "Upload a brain MRI image to generate an AI-based classification result.")

if  "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

uploaded_file = st.file_uploader( "Upload Brain MRI Image",  type=["jpg", "jpeg", "png"],
                                key =f"uploader_{st.session_state.uploader_key}")


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [  "glioma_tumor",    "meningioma_tumor",    "no_tumor",    "pituitary_tumor"]


# =========================================================
# IMAGE PROCESSING & PREDICTION
# =========================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(  '<div class="section-title">🖼️ MRI Scan Preview</div>',       unsafe_allow_html=True   )

    col1, col2 = st.columns([1.2, 1])

    # -----------------------------------------------------
    # IMAGE DISPLAY
    # -----------------------------------------------------

    with col1:

        st.image(     image,     caption="Uploaded MRI Image"  )

    # -----------------------------------------------------
    # IMAGE PROCESSING
    # -----------------------------------------------------

    img = image.resize((224, 224))

    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with st.spinner("🔍 Analyzing MRI image..."):

        prediction = model.predict(img, verbose=0)

    # Get probability values
    probabilities = prediction[0]

    # Predicted class
    predicted_index = int(np.argmax(probabilities))

    predicted_class = class_names[predicted_index]

    # Confidence
    confidence = float(np.max(probabilities)) * 100

    # Prediction Probability Analysis
st.subheader("📊 Prediction Probability Analysis")

probabilities = np.squeeze(prediction)

for class_name, probability in zip(class_names, probabilities):
    percentage = float(probability) * 100
    
    st.write(f"**{class_name.replace('_', ' ').title()} — {percentage:.2f}%**")
    st.progress(int(percentage))

    # Display result
    st.success(f" Prediction:{predicted1_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Analysis details
    analysis_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
    file_name = uploaded_file.name


    # =====================================================
    # RESULT
    # =====================================================

    with col2:

        st.markdown( '<div class="section-title">🔬 AI Analysis Result</div>',         unsafe_allow_html=True    )

        # Convert class name to professional text

        display_names = {
            "glioma_tumor": "Glioma Tumor",
            "meningioma_tumor": "Meningioma Tumor",
            "no_tumor": "No Tumor",
            "pituitary_tumor": "Pituitary Tumor"
        }

        result_name = display_names[predicted_class]

        if predicted_class == "no_tumor":

            st.success(
                f"### ✅ {result_name}"
            )

        else:

            st.warning(
                f"### ⚠️ {result_name}"
            )

        st.markdown(
            f"""
            <div class="result-card">

            <b>Predicted Class</b><br>
            {result_name}

            <br><br>

            <b>Confidence Score</b><br>
            {confidence:.2f}%

            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            min(confidence / 100, 1.0)
        )

        st.caption(  f"Model confidence: {confidence:.2f}%")
     
    # =====================================================
    # PROBABILITY ANALYSIS
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Prediction Probability Analysis</div>',
        unsafe_allow_html=True
    )

    probability_names = {
        "glioma_tumor": "Glioma Tumor",
        "meningioma_tumor": "Meningioma Tumor",
        "no_tumor": "No Tumor",
        "pituitary_tumor": "Pituitary Tumor"
    }

    for i, class_name in enumerate(class_names):

        probability = float(probabilities[i]) * 100

        st.write(
            f"**{probability_names[class_name]}** — "
            f"{probability:.2f}%"
        )

        st.progress(
            min(probability / 100, 1.0)
        )


    # =====================================================
    # IMAGE PROCESSING INFORMATION
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">⚙️ Image Processing</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="info-card">
            <b>Input Resolution</b><br>
            224 × 224 pixels
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
            <b>Normalization</b><br>
            Pixel values / 255
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="info-card">
            <b>Model Inference</b><br>
            Completed ✓
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # ABOUT MODEL
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🤖 About This AI System</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This application uses a trained deep learning model to classify
        brain MRI images into four categories: Glioma Tumor,
        Meningioma Tumor, Pituitary Tumor, and No Tumor.
        """
    )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.markdown("---")

    st.markdown(
        """
        <div class="disclaimer">

        ⚠️ <b>Medical Disclaimer</b><br><br>

        This application is developed for educational and research
        purposes. The AI prediction should not be considered a medical
        diagnosis or a substitute for professional medical advice.
        Please consult a qualified healthcare professional for clinical
        interpretation.

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    # =====================================================
    # INITIAL STATE
    # =====================================================

    st.info(
        "👆 Please upload a brain MRI image to start the analysis."
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">✨ How It Works</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="info-card">
            <h3>1️⃣ Upload</h3>
            Upload a brain MRI image.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
            <h3>2️⃣ Analyze</h3>
            AI model processes the MRI image.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="info-card">
            <h3>3️⃣ Result</h3>
            View prediction and confidence.
            </div>
            """,
            unsafe_allow_html=True
        )
