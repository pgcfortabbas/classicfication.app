import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
# Make sure the path matches where you saved the model in Google Drive
model_path = '/content/drive/MyDrive/saved_model.keras'
model = tf.keras.models.load_model(model_path)

# Define the image size the model was trained on
image_size = (128, 128) # Make sure this matches the size used during training

st.title("Image Classification App")

st.write("Upload an image and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file, target_size=image_size)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 # Add batch dimension and normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)

    # Assuming you have a way to map the class index back to the original class name
    # If you used LabelEncoder, you might need to save and load it as well
    # For now, let's just print the index and probability
    st.write(f"Prediction (class index): {predicted_class_index}")
    st.write(f"Prediction probability: {predictions[0][predicted_class_index]:.4f}")

    # If you saved the label_encoder, you can uncomment and use this:
    # Load the label encoder
    # import pickle
    # with open('/content/drive/MyDrive/label_encoder.pkl', 'rb') as f:
    #     label_encoder = pickle.load(f)
    # predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]
    # st.write(f"Predicted class: {predicted_class_name}")
