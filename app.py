import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import gdown
import os

DRIVE_FILE_ID = "1qlCZhwRvsQuJeSmRlHiQXYYqsUdR4h2q" 
MODEL_FILENAME = "saved_model.keras" 
image_size = (128, 128)
CLASS_NAMES = ["Cat", "Dog"]
CONFIDENCE_THRESHOLD = 0.75

@st.cache_resource
def download_and_load_model():
    try:
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner(f"Downloading model {MODEL_FILENAME} from Google Drive..."):
                gdown.download(id=DRIVE_FILE_ID, output=MODEL_FILENAME, quiet=False)
                st.success("Model downloaded successfully!")

        return tf.keras.models.load_model(MODEL_FILENAME)
    
    except Exception as e:
        st.error(
            f"""
            **MODEL LOAD FAILED!**
            Please check the following:
            1. Is the `DRIVE_FILE_ID` correct in the script?
            2. Is the file shared publicly ("Anyone with the link") on Google Drive?
            
            Error details: {e}
            """
        )
        st.stop()

st.title("Image Classification App")
model = download_and_load_model()
st.write("Upload an image and the model will predict its class.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=image_size)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    
    predicted_probability = predictions[0][predicted_class_index]
    
    if predicted_probability >= CONFIDENCE_THRESHOLD:
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        result_message = f"**{predicted_class_name}**"
        st.success(f"Confident Prediction: {predicted_class_name}")
    else:
        predicted_class_name = "Uncertain/Invalid"
        result_message = "**Uncertain/Invalid** (Confidence too low)"
        st.warning("Prediction confidence is low. This image may not be suitable for the model.")

    st.markdown("### Prediction Results")
    st.write(f"Predicted Class: {result_message}")
    st.write(f"Prediction (class index): **{predicted_class_index}**")
    st.write(f"Confidence (Max Probability): **{predicted_probability:.4f}** (Threshold: {CONFIDENCE_THRESHOLD})")
