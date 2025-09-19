import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random

# ==========================================
# TAB 1 — Railway Track Defect Detection (Your Code)
# ==========================================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🚄 Railway Safety Monitoring System")
tab1, tab2 = st.tabs(["🧠 Track Fault Detection", "📍 GPS & Sensor Monitoring"])

with tab1:
    st.header("Detect Defective Railway Tracks")
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

# ==========================================
# TAB 2 — GPS Tracking + Sensor Readings
# ==========================================
with tab2:
    st.header("Train GPS Tracking & Sensor Monitoring")

    # -------------------------
    # Simulated Train GPS data
    # -------------------------
    train_names = [f"Train_{i}" for i in range(1, 8)]
    locations = [
        "Chennai", "Madurai", "Coimbatore",
        "Trichy", "Salem", "Tirunelveli", "Erode"
    ]

    data = []
    for t, loc in zip(train_names, locations):
        km_marker = random.randint(0, 500)
        speed = random.randint(40, 120)
        data.append([t, loc, km_marker, speed])

    df = pd.DataFrame(data, columns=["Train", "Location", "KM_Marker", "Speed"])

    st.subheader("🚉 Current Train Status")
    st.dataframe(df)

    # -------------------------
    # Collision detection alerts
    # -------------------------
    alerts = []
    safe_distance = 30  # km
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if abs(df.loc[i,"KM_Marker"] - df.loc[j,"KM_Marker"]) < safe_distance:
                faster = df.loc[i] if df.loc[i,"Speed"] > df.loc[j,"Speed"] else df.loc[j]
                slower = df.loc[j] if df.loc[i,"Speed"] > df.loc[j,"Speed"] else df.loc[i]
                alerts.append(f"⚠️ {faster['Train']} must slow down near {slower['Train']}")

    st.subheader("📢 Collision Alerts")
    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ No collision risks detected")

    # -------------------------
    # Simulated Sensor readings (demo)
    # -------------------------
    st.subheader("📡 Sensor Readings (Demo)")
    sensor_data = {
        "Ultrasonic Distance Sensor": f"{random.randint(50, 300)} cm",
        "Vibration Sensor": f"{random.uniform(0.1, 1.2):.2f} g",
        "Temperature Sensor": f"{random.uniform(25, 40):.1f} °C",
        "Track Displacement Sensor": f"{random.uniform(0.0, 2.0):.2f} mm"
    }

    for name, value in sensor_data.items():
        st.info(f"**{name}**: {value}")

    # -------------------------
    # Train position graph
    # -------------------------
    st.subheader("📍 Train Speed vs Position")
    fig, ax = plt.subplots(figsize=(8,4))
    scatter = ax.scatter(df["KM_Marker"], df["Speed"], c=df["Speed"], cmap='viridis')
    for i, row in df.iterrows():
        ax.text(row["KM_Marker"], row["Speed"]+2, row["Train"], fontsize=8)
    ax.set_xlabel("Track Position (KM)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Train Speed vs Position")
    st.pyplot(fig)
