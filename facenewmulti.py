import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# ===========================
# 📦 Model Loader
# ===========================
@st.cache_resource
def load_models():
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = "models/inswapper_128.onnx"

    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.isfile(model_path):
        with st.spinner("🔽 Downloading FaceSwap model..."):
            r = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("✅ Model downloaded!")

    face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, providers=["CPUExecutionProvider"])

    return face_analyzer, swapper

# ===========================
# 🔄 Face Swapper
# ===========================
def swap_face(src_img, tgt_img, src_obj, tgt_obj, swapper):
    swapped = swapper.get(tgt_img.copy(), tgt_obj, src_obj, paste_back=True)
    return swapped

# ===========================
# 🖼️ Face Cropping Utility
# ===========================
def crop_faces(image, faces):
    face_images = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_crop = image[y1:y2, x1:x2]
        face_images.append(face_crop)
    return face_images

# ===========================
# 🎛️ Streamlit UI
# ===========================
st.set_page_config(layout="wide")
st.title("🧑‍🎤 Realistic Face Swap App (SimSwap + Multi-Face Picker)")

st.sidebar.header("📤 Upload Images")
src_file = st.sidebar.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"])
tgt_file = st.sidebar.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if src_file and tgt_file:
    src_img = np.array(Image.open(src_file).convert("RGB"))
    tgt_img = np.array(Image.open(tgt_file).convert("RGB"))

    st.subheader("📷 Preview")
    col1, col2 = st.columns(2)
    col1.image(src_img, caption="Source Image", use_column_width=True)
    col2.image(tgt_img, caption="Target Image", use_column_width=True)

    with st.spinner("🔍 Detecting faces..."):
        fa, swapper = load_models()
        src_faces = fa.get(src_img)
        tgt_faces = fa.get(tgt_img)

    if len(src_faces) == 0 or len(tgt_faces) == 0:
        st.error("❌ No face detected in one of the images.")
    else:
        # Show source faces
        st.sidebar.subheader("👤 Source Face")
        src_thumbs = crop_faces(src_img, src_faces)
        for i, face in enumerate(src_thumbs):
            st.sidebar.image(face, caption=f"Source #{i}", width=100)
        src_index = st.sidebar.selectbox("Select Source Face", range(len(src_faces)))

        # Show target faces
        st.sidebar.subheader("🎯 Target Faces")
        tgt_thumbs = crop_faces(tgt_img, tgt_faces)
        for i, face in enumerate(tgt_thumbs):
            st.sidebar.image(face, caption=f"Target #{i}", width=100)
        tgt_indices = st.sidebar.multiselect("Select Target Faces", range(len(tgt_faces)), default=list(range(len(tgt_faces))))

        if st.sidebar.button("🔄 Swap Selected Faces"):
            with st.spinner("Running Face Swap..."):
                result = tgt_img.copy()
                for i in tgt_indices:
                    result = swap_face(src_img, result, src_faces[src_index], tgt_faces[i], swapper)
                st.success("✅ Face swap completed!")
                st.image(result, caption="🎉 Final Swapped Image", use_column_width=True)
