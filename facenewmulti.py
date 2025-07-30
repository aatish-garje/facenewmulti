import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

st.set_page_config(layout="wide")
st.title("🔁 Multi-Face Swap App with Mapping")

# ================================
# 🧠 Load Models
# ================================

@st.cache_resource
def load_models():
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = "models/inswapper_128.onnx"
    os.makedirs("models", exist_ok=True)

    if not os.path.isfile(model_path):
        with st.spinner("⬇️ Downloading face swap model..."):
            r = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, providers=["CPUExecutionProvider"])
    return app, swapper

# ================================
# 🔍 Helper - Draw Faces and IDs
# ================================

def draw_face_ids(image, faces):
    img = image.copy()
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = list(map(int, face.bbox))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'ID {i}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img

def crop_face(image, bbox):
    x1, y1, x2, y2 = [int(i) for i in bbox]
    face = image[y1:y2, x1:x2]
    return face

# ================================
# 🔄 Swapping Function
# ================================

def swap_selected_faces(tgt_img, tgt_faces, src_faces, mapping, swapper):
    result = tgt_img.copy()
    for tgt_id, src_id in mapping.items():
        if tgt_id < len(tgt_faces) and src_id < len(src_faces):
            result = swapper.get(result, tgt_faces[tgt_id], src_faces[src_id], paste_back=True)
    return result

# ================================
# 📤 Upload Interface
# ================================

st.sidebar.subheader("📤 Upload Source Image(s)")
src_files = st.sidebar.file_uploader("Upload Source Faces", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.sidebar.subheader("🎯 Upload Target Image")
tgt_file = st.sidebar.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if src_files and tgt_file:
    # Load models
    app, swapper = load_models()

    # Load and process target
    tgt_img = np.array(Image.open(tgt_file).convert("RGB"))
    tgt_faces = app.get(tgt_img)

    if not tgt_faces:
        st.error("❌ No faces detected in target image.")
    else:
        st.subheader("🎯 Target Image with Face IDs")
        st.image(draw_face_ids(tgt_img, tgt_faces), caption="Target Face IDs", use_column_width=True)

        # Collect source faces from all uploaded images
        all_src_faces = []
        src_img_list = []

        st.subheader("🧑‍💼 Source Face IDs")

        for i, file in enumerate(src_files):
            src_img = np.array(Image.open(file).convert("RGB"))
            faces = app.get(src_img)
            if not faces:
                st.warning(f"No face detected in Source Image {i}")
                continue
            all_src_faces.extend(faces)
            st.image(draw_face_ids(src_img, faces), caption=f"Source {i} Face IDs", use_column_width=True)
            src_img_list.append((src_img, faces))

        if not all_src_faces:
            st.error("❌ No faces found in any source image.")
        else:
            st.subheader("🔁 Map Source Face ID ➝ Target Face ID")
            mapping = {}

            for tgt_id in range(len(tgt_faces)):
                src_id = st.selectbox(f"Source Face ID for Target Face ID {tgt_id}:", list(range(len(all_src_faces))), key=f"map_{tgt_id}")
                mapping[tgt_id] = src_id

            if st.button("🚀 Swap Faces Now"):
                with st.spinner("Swapping faces..."):
                    result = swap_selected_faces(tgt_img, tgt_faces, all_src_faces, mapping, swapper)
                    st.success("✅ Face swap completed!")
                    st.image(result, caption="🧠 Swapped Output", use_column_width=True)
