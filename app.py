import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

st.title("🧠 Brain Tumor Segmentation & Classification")
st.write("Unggah gambar MRI untuk mendeteksi keberadaan dan lokasi tumor otak.")

# 1. Load Model
@st.cache_resource
def load_my_model():
    # Pastikan file .h5 ada di folder yang sama dengan app.py
    return tf.keras.models.load_model('brain_tumor_segmentation_model.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 2. Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar MRI (format .tif, .jpg, .png)...", type=["tif", "jpg", "png"])

if uploaded_file is not None:
    # Preprocessing Gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    # Resize untuk input model (256x256)
    img_input = cv2.resize(img_original, (256, 256)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # 3. Prediksi
    with st.spinner('Menganalisis gambar...'):
        outputs = model.predict(img_input)
        pred_mask = outputs[0]
        pred_class = outputs[1]
    
    # Post-processing Mask
    mask = (pred_mask[0] > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (img_original.shape[1], img_original.shape[0]))
    
    # Hasil Diagnosis
    label = "POSITIVE (Tumor Terdeteksi)" if pred_class[0][0] > 0.5 else "NEGATIVE (Normal)"
    color = "red" if "POSITIVE" in label else "green"
    
    st.subheader(f"Hasil Diagnosis: :{color}[{label}]")
    
    # 4. Visualisasi Kolom
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img_rgb, caption="Citra MRI Asli", use_column_width=True)
        
    with col2:
        st.image(mask_resized, caption="Hasil Segmentasi (Mask)", use_column_width=True)
        
    with col3:
        # Membuat Overlay (Kontur Merah)
        overlay = img_rgb.copy()
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
        st.image(overlay, caption="Deteksi Lokasi Tumor", use_column_width=True)

    # Informasi Tambahan
    if "POSITIVE" in label:
        st.warning("Peringatan: Sistem mendeteksi adanya massa tumor. Segera konsultasikan dengan ahli radiologi.")
    else:
        st.success("Sistem tidak mendeteksi adanya massa tumor yang signifikan.")