import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/soil_model.h5")
    with open("model/class_indices.json", "r") as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_model()


with open("soil_to_crops.json", "r") as f:
    soil_to_crops = json.load(f)


st.title("🌾 Smart Crop Recommendation System")
st.write("Upload a soil image to identify its type and suggest suitable crops.")

uploaded_file = st.file_uploader("Upload soil image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Soil Image", use_container_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]

    st.subheader(f"🧭 Predicted Soil Type: **{predicted_class.title()} Soil**")

    if predicted_class in soil_to_crops:
        crops = soil_to_crops[predicted_class]
        st.success("🌱 Recommended Crops:")
        for crop in crops:
            st.write(f"- {crop}")
    else:
        st.warning("No crop data found for this soil type.")
st.markdown("""
<hr style="border:1px solid #ccc"/>

<div style="text-align:center; padding:8px;">
    <span style="font-size:18px; font-weight:bold; color:#1976d2;">
        Created by <span style="color:#f57c00;">VickyVignesh0913</span> 🚀<br>
        <span style="font-size:17px; color:#388e3c;"><b>Team: WHITE COLLARS</b></span>
    </span>
    <br>
    <span style="font-size:14px; color:#444;">
        Connect with me on <a href="https://github.com/VickyVignesh0913" target="_blank" style="color:#1976d2;">GitHub</a>
    </span>
</div>
""", unsafe_allow_html=True)


