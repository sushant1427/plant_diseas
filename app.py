import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
import os

# Function to download the model file if not present
def download_model(url, output_path):
    if not os.path.exists(output_path):
        with st.spinner('Downloading model... Please wait...'):
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success('Model downloaded successfully!')
            else:
                st.error(f'Failed to download the model file. Status code: {r.status_code}. Check the URL or network.')
                return False
    return True

# Set the path and URL for the model
model_path = 'cotton_disease_model.h5'
model_url = 'https://drive.google.com/uc?export=download&id=1zbnld7WSMWagF-n9eQBfiIsdCCwviOgY'

# Ensure model is downloaded
if download_model(model_url, model_path):
    # Load the trained model
    try:
        model = load_model(model_path)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.stop()

# Function to preprocess the image
# Function to preprocess the image
def preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)  # Resize the image to match model input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image to range [0, 1]
    return img_array


# Function to make predictions
def predict_disease(image_file):
    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Cotton Plant Disease Classification")
    st.write("Upload an image of a cotton plant or leaf to classify its disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        
        if st.button('Classify'):
            with st.spinner('Classifying...'):
                prediction = predict_disease(uploaded_file)
                disease_class = np.argmax(prediction)
                classes = ["Diseased cotton leaf", "Diseased cotton plant", "Fresh cotton leaf", "Fresh cotton plant"]
                st.write(f"Prediction: {classes[disease_class]}")

if __name__ == '__main__':
    main()
