import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model (.h5)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("railway_model.h5")
    return model

model = load_model()

# Class names (adjust if your dataset class order is reversed)
class_names = ["Non-Defective", "Defective"]

# App UI
st.title("🚂 Railway Track Fault Detection")
st.write("Upload a railway track image to check if it's defective or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = class_names[1] if prediction > 0.5 else class_names[0]

    st.markdown(f"### 🏁 Prediction: **{label}**")
