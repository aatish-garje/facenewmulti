import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# =============================
# Load ONNX Model and FaceAnalysis
# =============================
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

    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, providers=["CPUExecutionProvider"])
    return fa, swapper

# =============================
# Face Swapper
# =============================
def swap_faces_multi(src_img, tgt_img, fa, swapper, selected_face_index):
    src_faces = fa.get(src_img)
    tgt_faces = fa.get(tgt_img)

    if len(src_faces) == 0:
        return None, "❌ No face found in source image."
    if len(tgt_faces) == 0:
        return None, "❌ No face found in target image."

    src_face = src_faces[0]
    tgt_face = tgt_faces[selected_face_index]

    swapped = tgt_img.copy()
    swapped = swapper.get(swapped, tgt_face, src_face, paste_back=True)
    return swapped, None

# =============================
# Streamlit UI
# =============================
st.set_page_config(layout="wide")
st.title("😎 Multi-Face Streamlit Face Swap")

st.sidebar.header("📤 Upload Images")
src_file = st.sidebar.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"])
tgt_file = st.sidebar.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if src_file and tgt_file:
    src_img = np.array(Image.open(src_file).convert("RGB"))
    tgt_img = np.array(Image.open(tgt_file).convert("RGB"))

    st.subheader("📸 Preview")
    col1, col2 = st.columns(2)
    col1.image(src_img, caption="Source Face", use_container_width=True)
    col2.image(tgt_img, caption="Target Image", use_container_width=True)

    fa, swapper = load_models()
    tgt_faces = fa.get(tgt_img)

    if len(tgt_faces) == 0:
        st.error("❌ No face detected in target image.")
    else:
        st.markdown("### 🎯 Detected Faces in Target Image")
        preview_cols = st.columns(len(tgt_faces))
        for i, face in enumerate(tgt_faces):
            x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
            face_crop = tgt_img[y1:y2, x1:x2]
            if face_crop.size != 0:
                preview_cols[i].image(face_crop, caption=f"Target #{i}", width=100)

        selected_index = st.selectbox("👤 Select Face to Swap", list(range(len(tgt_faces))), index=0)

        if st.button("🔄 Swap Face"):
            with st.spinner("Swapping face..."):
                result, error = swap_faces_multi(src_img, tgt_img, fa, swapper, selected_index)
                if error:
                    st.error(error)
                else:
                    st.success("✅ Face Swapped Successfully!")
                    st.image(result, caption="🧠 Final Output", use_container_width=True)
