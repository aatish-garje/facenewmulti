import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# =========================
# 📥 Load ONNX Model
# =========================

@st.cache_resource
def load_models():
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = "models/inswapper_128.onnx"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.isfile(model_path):
        with st.spinner("📦 Downloading model..."):
            r = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, providers=["CPUExecutionProvider"])
    return app, swapper

# =========================
# 🔁 Face Swap Logic
# =========================

def swap_faces(target_img, source_faces, selected_map, app, swapper):
    tgt_faces = app.get(target_img)
    if not tgt_faces:
        return None, "❌ No faces found in target."

    result_img = target_img.copy()

    for tgt_idx, src_idx in selected_map.items():
        if tgt_idx >= len(tgt_faces) or src_idx >= len(source_faces):
            continue
        result_img = swapper.get(result_img, tgt_faces[tgt_idx], source_faces[src_idx], paste_back=True)

    return result_img, None

# =========================
# 🎛️ Streamlit UI
# =========================

st.set_page_config(layout="wide")
st.title("🧑‍🤝‍🧑 Multi-Face Swap App (ONNX + InsightFace)")

app, swapper = load_models()

st.sidebar.header("📤 Upload Target Image (with multiple faces)")
tgt_file = st.sidebar.file_uploader("Target Image", type=["jpg", "jpeg", "png"])

st.sidebar.header("📥 Upload Source Face(s)")
src_files = st.sidebar.file_uploader("Source Faces (can upload multiple)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if tgt_file and src_files:
    tgt_img = np.array(Image.open(tgt_file).convert("RGB"))
    src_imgs = [np.array(Image.open(f).convert("RGB")) for f in src_files]

    tgt_faces = app.get(tgt_img)
    src_faces = []
    for img in src_imgs:
        detected = app.get(img)
        if detected:
            src_faces.append(detected[0])

    st.subheader("👥 Detected Faces")
    st.markdown("### 🧑‍🎯 Target Faces")
    tgt_cols = st.columns(len(tgt_faces))
    for i, face in enumerate(tgt_faces):
        x1, y1, x2, y2 = map(int, face.bbox)
        face_crop = tgt_img[y1:y2, x1:x2]
        tgt_cols[i].image(face_crop, caption=f"Target #{i}", width=120)

    st.markdown("### 🧑 Source Faces")
    src_cols = st.columns(len(src_faces))
    for i, face in enumerate(src_faces):
        face_crop = src_imgs[i][face.bbox[1]:face.bbox[3], face.bbox[0]:face.bbox[2]]
        src_cols[i].image(face_crop, caption=f"Source #{i}", width=120)

    st.markdown("### 🔁 Select Swaps")
    selected_map = {}
    for i in range(len(tgt_faces)):
        src_idx = st.selectbox(f"Replace Target #{i} with Source Face:", [-1] + list(range(len(src_faces))), key=f"select_{i}")
        if src_idx != -1:
            selected_map[i] = src_idx

    if st.button("🔄 Run Face Swap"):
        with st.spinner("Running..."):
            result, err = swap_faces(tgt_img, src_faces, selected_map, app, swapper)
            if err:
                st.error(err)
            else:
                st.success("✅ Swap Complete!")
                st.image(result, caption="🎯 Final Output", use_column_width=True)
