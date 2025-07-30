import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# =========================
# Load ONNX Model & Detector
# =========================
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

# =========================
# Swap Faces
# =========================
def swap_faces(src_img, tgt_img, fa, swapper, src_face_index, tgt_indices):
    src_faces = fa.get(src_img)
    tgt_faces = fa.get(tgt_img)

    if len(src_faces) == 0:
        return None, "❌ No face in source."
    if len(tgt_faces) == 0:
        return None, "❌ No face in target."

    swapped = tgt_img.copy()

    src_face = src_faces[src_face_index]
    for i in tgt_indices:
        swapped = swapper.get(swapped, tgt_faces[i], src_face, paste_back=True)

    return swapped, None

# =========================
# Streamlit App
# =========================
st.set_page_config(layout="wide")
st.title("🧑‍🤝‍🧑 Multi-Face Swap App (SimSwap ONNX)")

st.sidebar.header("📤 Upload Images")
src_file = st.sidebar.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
tgt_file = st.sidebar.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if src_file and tgt_file:
    src_img = np.array(Image.open(src_file).convert("RGB"))
    tgt_img = np.array(Image.open(tgt_file).convert("RGB"))

    st.subheader("📷 Preview Images")
    col1, col2 = st.columns(2)
    col1.image(src_img, caption="Source Image", use_column_width=True)
    col2.image(tgt_img, caption="Target Image", use_column_width=True)

    with st.spinner("🔍 Detecting Faces..."):
        fa, swapper = load_models()
        src_faces = fa.get(src_img)
        tgt_faces = fa.get(tgt_img)

    st.sidebar.markdown(f"🧑 Found `{len(src_faces)}` face(s) in source")
    st.sidebar.markdown(f"🧑 Found `{len(tgt_faces)}` face(s) in target")

    src_index = st.sidebar.selectbox("Select Source Face", range(len(src_faces)))
    tgt_index = st.sidebar.multiselect("Select Target Face(s) to Replace", range(len(tgt_faces)), default=list(range(len(tgt_faces))))

    if st.sidebar.button("🔄 Swap Selected Faces"):
        with st.spinner("Swapping..."):
            result, err = swap_faces(src_img, tgt_img, fa, swapper, src_index, tgt_index)
            if err:
                st.error(err)
            else:
                st.image(result, caption="🎯 Face Swap Result", use_column_width=True)
