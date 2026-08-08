import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]


display_names = {
    "glioma_tumor": "Glioma Tumor",
    "meningioma_tumor": "Meningioma Tumor",
    "no_tumor": "No Tumor",
    "pituitary_tumor": "Pituitary Tumor"
}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🤖 Model Information")

    st.write("**Model:** Deep Learning")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Input Size:** 224 × 224")
    st.write("**Number of Classes:** 4")

    st.divider()

    st.markdown("## 🧠 Classification Classes")

    st.write("• Glioma Tumor")
    st.write("• Meningioma Tumor")
    st.write("• Pituitary Tumor")
    st.write("• No Tumor")

    st.divider()

    if model_loaded:
        st.success("🟢 Model Loaded")
    else:
        st.error("🔴 Model Not Loaded")


# =========================================================
# MAIN TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🧠 Brain Tumor Detection AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Deep Learning Based Brain MRI Image Classification System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload MRI Scan</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a brain MRI image to generate an AI-based classification result."
)


uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# HOW IT WORKS
# =========================================================

if uploaded_file is None:

    st.info("👋 Please upload a brain MRI image to start the analysis.")

    st.divider()

    st.markdown(
        '<div class="section-title">✨ How It Works</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Upload")
        st.write("Upload a brain MRI image.")

    with col2:
        st.markdown("### 2️⃣ Analyze")
        st.write("AI model processes the MRI image.")

    with col3:
        st.markdown("### 3️⃣ Result")
        st.write("View prediction and confidence.")

    st.divider()

    st.markdown(
        '<div class="section-title">🤖 About This AI System</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This application uses a trained deep learning model "
        "to classify brain MRI images into four categories: "
        "Glioma Tumor, Meningioma Tumor, Pituitary Tumor, "
        "and No Tumor."
    )

    st.warning(
        "⚠️ Medical Disclaimer: This application is developed "
        "for educational and research purposes. The AI prediction "
        "should not be considered a medical diagnosis or a "
        "substitute for professional medical advice. "
        "Please consult a qualified healthcare professional "
        "for clinical interpretation."
    )


# =========================================================
# IMAGE ANALYSIS
# =========================================================

if uploaded_file is not None and model_loaded:

    try:

        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        image = Image.open(uploaded_file).convert("RGB")


        # -------------------------------------------------
        # IMAGE PREVIEW
        # -------------------------------------------------

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="section-title">🖼️ MRI Scan Preview</div>',
                unsafe_allow_html=True
            )

            st.image(image,caption="Uploaded MRI Image")


        # -------------------------------------------------
        # IMAGE PREPROCESSING
        # -------------------------------------------------

        img = image.resize((224, 224))

        img_array = np.array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(img_array, verbose=0)

        probabilities = np.squeeze(prediction)


        # -------------------------------------------------
        # MAKE SURE PROBABILITIES ARE VALID
        # -------------------------------------------------

        probabilities = np.asarray(probabilities, dtype=np.float32)

        if probabilities.ndim != 1:
            probabilities = probabilities.flatten()


        # If model output is not probability distribution,
        # convert it using softmax.

        if (
            np.any(probabilities < 0)
            or not np.isclose(np.sum(probabilities), 1.0, atol=0.01)
        ):

            probabilities = tf.nn.softmax(
                probabilities
            ).numpy()


        # -------------------------------------------------
        # PREDICTED CLASS
        # -------------------------------------------------

        predicted_index = int(np.argmax(probabilities))

        predicted_class = class_names[predicted_index]

        result_name = display_names[predicted_class]


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = float(
            np.max(probabilities) * 100
        )


        # -------------------------------------------------
        # RESULT DISPLAY
        # -------------------------------------------------

        with col2:

            st.markdown(
                '<div class="section-title">🔬 AI Analysis Result</div>',
                unsafe_allow_html=True
            )


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

                <b>Predicted Class</b>
                <br><br>

                {result_name}

                <br><br><br>

                <b>Confidence Score</b>
                <br><br>

                {confidence:.2f}%

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(int(confidence), 100)
            )

            st.caption(
                f"Model confidence: {confidence:.2f}%"
            )


        # =================================================
        # PREDICTION PROBABILITY ANALYSIS
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Prediction Probability Analysis</div>',
            unsafe_allow_html=True
        )


        for class_name, probability in zip(
            class_names,
            probabilities
        ):

            percentage = float(probability * 100)

            professional_name = display_names[class_name]

            st.write(
                f"**{professional_name} — {percentage:.2f}%**"
            )

            st.progress(
                min(int(percentage), 100)
            )


        # =================================================
        # IMAGE PROCESSING INFORMATION
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">⚙️ Image Processing</div>',
            unsafe_allow_html=True
        )


        info1, info2, info3 = st.columns(3)


        with info1:

            st.markdown("### Input Resolution")

            st.write("224 × 224 pixels")


        with info2:

            st.markdown("### Normalization")

            st.write("Pixel values / 255")


        with info3:

            st.markdown("### Model Inference")

            st.write("Completed ✓")


        # =================================================
        # ANALYSIS DETAILS
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">📋 Analysis Details</div>',
            unsafe_allow_html=True
        )


        analysis_time = datetime.now().strftime(
            "%d %b %Y, %I:%M %p"
        )

        file_name = uploaded_file.name


        detail1, detail2 = st.columns(2)


        with detail1:

            st.write(
                f"**File Name:** {file_name}"
            )

            st.write(
                f"**Analysis Time:** {analysis_time}"
            )


        with detail2:

            st.write(
                "**Model:** Deep Learning"
            )

            st.write(
                "**Framework:** TensorFlow / Keras"
            )


        # =================================================
        # ABOUT SYSTEM
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">🤖 About This AI System</div>',
            unsafe_allow_html=True
        )

        st.write(
            "This application uses a trained deep learning model "
            "to classify brain MRI images into four categories: "
            "Glioma Tumor, Meningioma Tumor, Pituitary Tumor, "
            "and No Tumor."
        )


        # =================================================
        # MEDICAL DISCLAIMER
        # =================================================

        st.warning(
            "⚠️ Medical Disclaimer: This application is developed "
            "for educational and research purposes. The prediction "
            "generated by this AI model should not be considered "
            "a medical diagnosis or a substitute for professional "
            "medical advice. Please consult a qualified healthcare "
            "professional for clinical interpretation."
        )


    except Exception as e:

        st.error(
            "❌ An error occurred while processing the MRI image."
        )

        st.write(
            "Please make sure the uploaded file is a valid "
            "JPG, JPEG, or PNG MRI image."
        )

        st.exception(e)


# =========================================================
# MODEL ERROR
# =========================================================

elif uploaded_file is not None and not model_loaded:

    st.error(
        "❌ The trained model could not be loaded."
    )

    st.info(
        "Please make sure that 'brain_tumor_model.keras' "
        "is present in the GitHub repository."
    )
