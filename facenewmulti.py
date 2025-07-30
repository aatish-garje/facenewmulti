import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# -------------------------------
# Load ONNX Face Swap Model
# -------------------------------
@st.cache_resource
def load_models():
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = "models/inswapper_128.onnx"
    os.makedirs("models", exist_ok=True)

    if not os.path.isfile(model_path):
        with st.spinner("🔽 Downloading FaceSwap model..."):
            r = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("✅ Model downloaded!")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, providers=["CPUExecutionProvider"])
    return app, swapper

# -------------------------------
# Extract faces and assign IDs
# -------------------------------
def extract_faces(img_list, app, prefix):
    faces = []
    face_images = []
    for idx, img in enumerate(img_list):
        detected = app.get(img)
        for face in detected:
            x1, y1, x2, y2 = [int(i) for i in face.bbox]
            face_crop = img[y1:y2, x1:x2]
            face_images.append(face_crop)
            faces.append({"face": face, "img": img, "id": f"{prefix}{len(faces)}"})
    return face_images, faces

# -------------------------------
# Perform Face Swapping
# -------------------------------
def apply_swap(tgt_img, target_faces, mapping, source_faces, swapper):
    output = tgt_img.copy()
    for tgt_idx, src_id in mapping.items():
        if src_id == "skip":
            continue
        src = next((f for f in source_faces if f["id"] == src_id), None)
        if src is None:
            continue
        tgt = target_faces[tgt_idx]["face"]
        output = swapper.get(output, tgt, src["face"], paste_back=True)
    return output

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("🤖 Multi-Face Swap App (SimSwap + InsightFace)")

app, swapper = load_models()

st.sidebar.header("Upload Images")
src_files = st.sidebar.file_uploader("Upload One or More Source Face Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
tgt_file = st.sidebar.file_uploader("Upload Target Image (with Multiple Faces)", type=["jpg", "jpeg", "png"])

if src_files and tgt_file:
    src_imgs = [np.array(Image.open(f).convert("RGB")) for f in src_files]
    tgt_img = np.array(Image.open(tgt_file).convert("RGB"))

    st.subheader("📸 Previews")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧑 Source Faces")
        src_faces_img, src_faces_data = extract_faces(src_imgs, app, "S")
        for i, face in enumerate(src_faces_img):
            st.image(face, caption=f"Source ID: {src_faces_data[i]['id']}", width=150)

    with col2:
        st.markdown("### 🎯 Target Faces")
        tgt_faces_img, tgt_faces_data = extract_faces([tgt_img], app, "T")
        for i, face in enumerate(tgt_faces_img):
            st.image(face, caption=f"Target ID: {i}", width=150)

    # -------------------------------
    # Build Face Mapping Interface
    # -------------------------------
    st.subheader("🔁 Map Source → Target Faces")
    face_map = {}
    src_ids = [s["id"] for s in src_faces_data]
    src_ids_with_skip = ["skip"] + src_ids

    for i in range(len(tgt_faces_data)):
        selected = st.selectbox(f"Target Face {i}: Swap with?", options=src_ids_with_skip, key=f"map_{i}")
        face_map[i] = selected

    if st.button("🔄 Perform Face Swap"):
        with st.spinner("Processing face swaps..."):
            result = apply_swap(tgt_img, tgt_faces_data, face_map, src_faces_data, swapper)
            st.success("✅ Swap completed!")
            st.image(result, caption="🔍 Final Output", use_column_width=True)
