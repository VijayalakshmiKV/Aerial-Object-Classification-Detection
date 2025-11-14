import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Bird vs Drone Classifier", layout="centered")
st.title("Bird vs Drone Image Classifier")
st.write("Upload an image and the model will classify it as **Bird** or **Drone**.")


# -------------------------------------------------
# 2. Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_custom_cnn_model.h5")

model = load_model()


# -------------------------------------------------
# 3. File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -------------------------------------------------
    # 4. Preprocess the image
    # -------------------------------------------------
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # -------------------------------------------------
    # 5. Predict (binary sigmoid model)
    # -------------------------------------------------
    raw_pred = model.predict(img_array)[0][0]   # Single output neuron

    # Convert sigmoid output to class
    if raw_pred >= 0.5:
        predicted_label = "drone"
        confidence = raw_pred
    else:
        predicted_label = "bird"
        confidence = 1 - raw_pred

    # -------------------------------------------------
    # 6. Display the results
    # -------------------------------------------------
    st.markdown("###  Prediction Results")
    st.write(f"**Predicted Class:** {predicted_label.upper()}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # Debugging info
    st.write("Raw sigmoid output:", float(raw_pred))

    # Confidence bar
    st.progress(int(confidence * 100))

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(" Streamlit and TensorFlow")
