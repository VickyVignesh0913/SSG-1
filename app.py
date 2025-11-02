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

st.markdown("""
<div style="background:#e8f5e9; border-radius:6px; padding:8px; margin-bottom:10px;">
    <b>🖼️ Want to try? Download a sample soil image below!</b>
</div>
""", unsafe_allow_html=True)

with open("sample_soil.jpg", "rb") as file:
    st.download_button(label="Download Sample Soil Image",
                      data=file,
                      file_name="sample_soil.jpg",
                      mime="image/jpeg")


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
<hr style="border-top:2px solid #1976d2; margin-top:36px;">

<div style="max-width:450px; margin:28px auto;">
  <div style="background:#f5f7fa; border-radius:12px; box-shadow:0 2px 10px #e0e7ef; padding:18px 24px; text-align:center; border:1px solid #1976d2;">
    <span style="font-size:22px; font-weight:800; color:#1976d2;">
      🚀 Developed by <span style="color:#f57c00;">VickyVignesh0913</span>
    </span>
    <br>
    <span style="font-size:16px; color:#388e3c; font-weight:700;">
      Team: <span style="color:#004d40;">WHITE COLLARS</span>
    </span>
    <hr style="border:1px dashed #1976d2; margin:10px 0;">
    <span style="font-size:15px; color:#444;">
      <b>Connect:</b>
      <a href="https://github.com/VickyVignesh0913" target="_blank" style="color:#1976d2;">GitHub</a> |
      <a href="mailto:yourmail@example.com" style="color:#d84315;">Mail</a>
    </span>
  </div>
  <div style="text-align:center; font-size:13px; color:#888; margin-top:12px;">
    &copy; 2025 <b>Smart Crop Recommender</b>
  </div>
</div>
""", unsafe_allow_html=True)



