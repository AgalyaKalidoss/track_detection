import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Railway Track Defect Detection")
st.write("Upload an image of a railway track to check if it's defective or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Show raw score (optional)
    st.write("🔍 Raw output:", float(prediction))

    # Custom threshold
    threshold = 0.462  # Adjusted empirically
    if prediction < threshold:
        st.error("⚠️ Defective Track Detected")
    else:
        st.success("✅ Track is Properly Aligned")
