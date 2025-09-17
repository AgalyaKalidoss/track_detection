import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===== Load TFLite model =====
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="railway_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

# ===== Prediction function =====
def predict_tflite(img_array):
    img_array = img_array.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]

# ===== UI =====
st.title("Railway Track Defect Detection")
st.write("Upload an image of a railway track to check if it's defective or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = predict_tflite(img_array)

    if prediction > 0.5:
        st.error("⚠️ Defective Track Detected")
    else:
        st.success("✅ Track is Properly Aligned")
