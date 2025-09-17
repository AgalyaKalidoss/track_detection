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
    st.image(img, caption="Uploaded Image", use_container_width=True)  # ✅ updated

    # Preprocess
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Debug: see actual output
    st.write("Raw output:", output.tolist())

    # Decide based on output shape
    if output.shape == ():  # single value
        score = output
    else:
        score = output[0] if output.shape[0] == 1 else output

    # If binary (single value)
    if len(output.shape) == 0 or output.shape[0] == 1:
        if score > 0.5:
            st.error("⚠️ Defective Track Detected")
        else:
            st.success("✅ Track is Properly Aligned")
    else:
        # If 2-class softmax output
        predicted_class = np.argmax(output)
        if predicted_class == 1:
            st.error("⚠️ Defective Track Detected")
        else:
            st.success("✅ Track is Properly Aligned")
