import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import gdown
import os

# --- Configuration for Google Drive Download ---
# 1. Find the Keras file (saved_model.keras) in your Google Drive.
# 2. Right-click it, choose "Share," and change access to "Anyone with the link."
# 3. Copy the URL and extract the unique File ID (the string between /d/ and /view).
# 4. PASTE that ID below:
DRIVE_FILE_ID = "1qlCZhwRvsQuJeSmRlHiQXYYqsUdR4h2q" 

# Define the local filename where the model will be saved on the Streamlit server
MODEL_FILENAME = "saved_model.keras" 

# Define the image size the model was trained on
image_size = (128, 128) # Make sure this matches the size used during training

# --- Model Loading with Caching ---
# @st.cache_resource ensures the model is downloaded and loaded only once,
# which is crucial for fast deployment and good performance.
@st.cache_resource
def download_and_load_model():
    """Downloads the Keras model from Google Drive and loads it into memory."""
    try:
        # 1. Check if the model file already exists locally (after the first run)
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner(f"Downloading model {MODEL_FILENAME} from Google Drive..."):
                # Use gdown to download the file using the ID
                gdown.download(id=DRIVE_FILE_ID, output=MODEL_FILENAME, quiet=False)
                st.success("Model downloaded successfully!")

        # 2. Load the model from the local path
        return tf.keras.models.load_model(MODEL_FILENAME)
    
    except Exception as e:
        # Provide a specific error message if the download or load fails
        st.error(
            f"""
            **MODEL LOAD FAILED!**
            Please check the following:
            1. Is the `DRIVE_FILE_ID` correct in the script?
            2. Is the file shared publicly ("Anyone with the link") on Google Drive?
            
            Error details: {e}
            """
        )
        st.stop() # Stop the app execution

# Load the model once at startup
st.title("Image Classification App")
model = download_and_load_model()
st.write("Upload an image and the model will predict its class.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    # Note: load_img automatically handles resizing to image_size
    image = load_img(uploaded_file, target_size=image_size)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 # Add batch dimension and normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)

    # Output results
    st.markdown("### Prediction Results")
    st.write(f"Prediction (class index): **{predicted_class_index}**")
    st.write(f"Prediction probability: **{predictions[0][predicted_class_index]:.4f}**")

    # If you have class names, you can replace the index with them here:
    # class_names = ["cat", "dog", "bird"] 
    # st.write(f"Predicted class: {class_names[predicted_class_index]}")

    # If you saved the label_encoder, you can uncomment and use this:
    # import pickle
    # LE_FILE = "label_encoder.pkl"
    # gdown.download(id="LE_DRIVE_ID", output=LE_FILE, quiet=True)
    # with open(LE_FILE, 'rb') as f:
    #     label_encoder = pickle.load(f)
    # predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]
    # st.write(f"Predicted class: {predicted_class_name}")
