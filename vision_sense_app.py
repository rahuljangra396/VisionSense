"""
VisionSense - Streamlit AI Image Analyzer (uses YOLOv8-large)

Features:
- Upload image or capture from camera
- Object detection (YOLOv8 large, file: yolov8l.pt)
- OCR (pytesseract)
- Face detection (OpenCV Haar cascade)
- Optional AI explanation using OpenAI (if OPENAI_API_KEY in .env)
- Download annotated image
"""

import os
import io
import base64
import tempfile
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import pytesseract
from ultralytics import YOLO
from dotenv import load_dotenv
import openai

# Load .env (if present)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="VisionSense", layout="wide", page_icon="🧠")
st.title("🧠 VisionSense — AI Image Analyzer")
st.caption("Object detection (YOLOv8), OCR, face detection, and optional AI explanation.")

# --- Paths & model ---
MODEL_FILE = "yolov8l.pt"  # your large YOLO model (you placed this in project folder)
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        return None
    model = YOLO(path)
    return model

yolo_model = load_yolo_model(MODEL_FILE)
if yolo_model is None:
    st.error(f"YOLO model not found at `{MODEL_FILE}`. Put the file in this folder.")
    st.stop()

# --- Left panel controls ---
with st.sidebar:
    st.header("Input")
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    use_camera = st.button("Use camera (take photo)")
    st.markdown("---")
    st.header("Modes")
    mode = st.radio("Choose analysis mode:", 
                    ("Object Detection", "Text (OCR)", "Face Detection", "AI Explanation"))
    st.markdown("---")
    st.header("YOLO Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25)
    iou = st.slider("NMS IoU threshold", 0.01, 0.9, 0.45)
    max_det = st.number_input("Max detections", min_value=1, max_value=500, value=100)
    st.markdown("---")
    st.write("Model file:", MODEL_FILE)
    st.write("OpenAI:", "Configured" if OPENAI_API_KEY else "Not configured (optional)")

# --- Helper utilities ---

def read_image_from_upload(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)  # RGB array
    except Exception as e:
        st.error("Failed to read image: " + str(e))
        return None

def to_pil(img_np):
    """ Convert BGR or RGB numpy array to PIL Image (RGB) """
    if img_np is None:
        return None
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        return Image.fromarray(img_np.astype('uint8'))
    else:
        return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype('uint8'))

def download_button(img_pil, filename="annotated.png"):
    """Return a link to download PIL image"""
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    b64 = base64.b64encode(byte_im).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">⬇ Download annotated image</a>'
    st.markdown(href, unsafe_allow_html=True)

def run_yolo_on_image(img_rgb, conf_thres=0.25, iou_thres=0.45, max_det=100):
    """
    img_rgb: numpy array in RGB
    Returns: annotated_image_rgb (numpy), detections_df (pd.DataFrame)
    """
    # ultralytics YOLO expects PIL/np image (RGB) or path
    results = yolo_model.predict(source=img_rgb, conf=conf_thres, iou=iou_thres, verbose=False, max_det=max_det)
    # Render annotated image (results[0].plot() returns np array RGB) - use .plot() or .plot() depending version
    try:
        annotated = results[0].plot()  # returns ndarray (RGB)
    except Exception:
        # fallback to render()
        annotated = np.squeeze(results.render())  # BGR? Usually RGB but ensure conversion
        # ultralytics render returns list of arrays — we took first
    # Get pandas df for detections
    try:
        df = results.pandas().xyxy[0]  # x1, y1, x2, y2, confidence, class, name
    except Exception:
        # Build manually
        df = pd.DataFrame()
    return annotated, df

def detect_faces_and_annotate(img_rgb):
    """Detect faces with Haar cascade and annotate (draw boxes)."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(HAAR_FILE)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    out = img_bgr.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(out, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out_rgb, faces

def ocr_image(img_rgb):
    """Run Tesseract OCR on image (expects RGB numpy array)."""
    pil = to_pil(img_rgb)
    text = pytesseract.image_to_string(pil)
    return text

def ai_explain_image(detections_df, ocr_text_sample):
    """
    Create a short prompt using detections and OCR text and ask OpenAI to explain.
    This avoids sending raw binary image to OpenAI (keeps it simple).
    """
    if not OPENAI_API_KEY:
        return "OpenAI key not configured. Add OPENAI_API_KEY to .env to enable AI explanations."

    # Build summary of detections
    det_summary = []
    if detections_df is not None and not detections_df.empty:
        # use top 6 detections by confidence
        top = detections_df.sort_values("confidence", ascending=False).head(6)
        for _, row in top.iterrows():
            det_summary.append(f"{row['name']} (conf {row['confidence']:.2f})")
    det_text = "; ".join(det_summary) if det_summary else "No objects detected."

    prompt = (
        f"I have an image. Detected objects: {det_text}.\n\n"
        f"Extracted text (if any): {ocr_text_sample[:500] or 'None'}.\n\n"
        "Based on the detected objects and any text, give a concise, human friendly description of what might be happening in the image, "
        "possible context (e.g. 'outdoor street scene', 'kitchen', 'office meeting'), and 2 short suggestions how someone could use this image (e.g. 'crop to show person')."
    )
    try:
        # Use ChatCompletion (gpt-3.5-turbo)
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=250,
            temperature=0.6
        )
        out = resp.choices[0].message.content.strip()
        return out
    except Exception as e:
        return f"OpenAI call failed: {e}"

# --- Main flow ---

img_rgb = None
source_desc = None

# If camera pressed, show camera input widget
if use_camera:
    cam_img = st.camera_input("Take a photo")
    if cam_img is not None:
        img_rgb = read_image_from_upload(cam_img)
        source_desc = "camera"
# Otherwise uploaded file
if upload is not None:
    img_rgb = read_image_from_upload(upload)
    source_desc = "upload"

if img_rgb is None:
    st.info("Upload an image or use the camera to start.")
    st.stop()

# Display input image
st.subheader("Input Image")
st.image(img_rgb, use_column_width=True)

# Run according to mode
if mode == "Object Detection":
    st.subheader("🔎 Object Detection (YOLOv8)")
    try:
        annotated, df = run_yolo_on_image(img_rgb, conf_thres=conf, iou_thres=iou, max_det=max_det)
        # Annotated may be RGB ndarray
        st.image(annotated, caption="Annotated (YOLO)", use_column_width=True)
        # Show detections table
        if df is not None and not df.empty:
            st.markdown("**Detections**")
            display_df = df[["name","confidence","xmin","ymin","xmax","ymax"]] if all(col in df.columns for col in ["name","confidence","xmin","ymin","xmax","ymax"]) else df
            st.dataframe(display_df.rename(columns={"xmin":"x1","ymin":"y1","xmax":"x2","ymax":"y2"}))
        else:
            st.info("No detections found.")
        # Download button
        pil_annotated = to_pil(annotated)
        download_button(pil_annotated, filename="vision_annotated.png")
    except Exception as e:
        st.error("Object detection failed: " + str(e))

elif mode == "Text (OCR)":
    st.subheader("🔤 Text Extraction (OCR)")
    try:
        text = ocr_image(img_rgb)
        if not text.strip():
            st.info("No text detected.")
        else:
            st.text_area("OCR Output", value=text, height=300)
        # Show small annotated image with boxes from OCR? (skip for simplicity)
    except Exception as e:
        st.error("OCR failed: " + str(e))

elif mode == "Face Detection":
    st.subheader("🙂 Face Detection")
    try:
        annotated_img, faces = detect_faces_and_annotate(img_rgb)
        st.image(annotated_img, caption=f"Faces found: {len(faces)}", use_column_width=True)
        if len(faces) == 0:
            st.info("No faces detected.")
        else:
            st.write("Face boxes (x, y, w, h):")
            st.write(faces.tolist() if isinstance(faces, np.ndarray) else faces)
        download_button(to_pil(annotated_img), filename="vision_faces.png")
    except Exception as e:
        st.error("Face detection failed: " + str(e))

elif mode == "AI Explanation":
    st.subheader("🤖 AI Image Explanation (summary from OpenAI)")
    try:
        # Run both YOLO and OCR to provide context
        annotated, df = run_yolo_on_image(img_rgb, conf_thres=conf, iou_thres=iou, max_det=max_det)
        ocr_text = ocr_image(img_rgb)
        explanation = ai_explain_image(df, ocr_text)
        st.markdown("**AI Explanation**")
        st.write(explanation)
        if df is not None and not df.empty:
            st.markdown("**Detections found (top 6)**")
            st.dataframe(df[["name","confidence"]].sort_values("confidence", ascending=False).head(6))
        if ocr_text.strip():
            st.markdown("**OCR (snippet)**")
            st.write(ocr_text[:800] + ("..." if len(ocr_text)>800 else ""))
        # show annotated
        st.image(annotated, caption="Annotated (YOLO)", use_column_width=True)
        download_button(to_pil(annotated), filename="vision_explain_annotated.png")
    except Exception as e:
        st.error("AI explanation failed: " + str(e))

st.markdown("---")
st.caption("VisionSense — YOLOv8 object detection, Tesseract OCR, and optional OpenAI explanation. Model file should be yolov8l.pt in the app folder.")
