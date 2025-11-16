import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import os
import time
from pathlib import Path
from twilio.rest import Client

# ---------------------------
# Helpers
# ---------------------------

def load_model(model_path: str | None):
    """
    Load a YOLO model.
    If model_path is None or does not exist, load yolov8n (pretrained) if available.
    """
    if model_path and Path(model_path).exists():
        st.sidebar.write(f"Loading model from `{model_path}`")
        model = YOLO(model_path)
    else:
        st.sidebar.write("Using default pretrained `yolov8n` model (will be downloaded if needed).")
        model = YOLO("yolov8n.pt")
    return model

def pil_to_numpy(pil_img: Image.Image):
    """Convert PIL image to numpy array (RGB)"""
    return np.array(pil_img.convert("RGB"))

def run_inference_on_image(model, image: Image.Image, conf=0.25):
    """Run model.predict on a PIL image and return annotated numpy image and result object."""
    img_np = pil_to_numpy(image)
    results = model.predict(source=img_np, conf=conf, save=False, verbose=False)
    # results[0].plot() returns annotated numpy array
    annotated = results[0].plot()  # numpy array (BGR or RGB depending on ultralytics version)
    # Ensure it's RGB for PIL/Streamlit (Ultralytics usually returns RGB)
    return annotated, results[0]

def find_latest_run_detect_dir(runs_root="runs/detect"):
    """Return the latest runs/detect/* directory path or None."""
    root = Path(runs_root)
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest

def find_saved_outputs_in_run(run_dir: Path, uploaded_filename: str | None = None):
    """
    Find files created by YOLO in the run directory. If uploaded_filename is provided,
    try to find a file containing its stem.
    """
    if not run_dir or not run_dir.exists():
        return []
    # Search inside run_dir and its subfolders
    files = list(run_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    if uploaded_filename:
        stem = Path(uploaded_filename).stem
        # prefer files that contain the stem
        filtered = [f for f in files if stem in f.name]
        if filtered:
            return filtered
    # fallback: return all files sorted by modification time (newest first)
    files_sorted = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
    return files_sorted

def save_temp_uploaded_file(uploaded_file) -> str:
    """Save uploaded_file (streamlit UploadedFile) into a temp file and return its path"""
    suffix = Path(uploaded_file.name).suffix if uploaded_file else ""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.read())
    tf.flush()
    tf.close()
    return tf.name

def send_sms_twilio(body: str):
    """Send SMS using Twilio if env vars are set. Return True if sent, False otherwise."""
    sid = os.environ.get("TWILIO_SID")
    token = os.environ.get("TWILIO_TOKEN")
    from_num = os.environ.get("TWILIO_FROM")
    to_num = os.environ.get("ALERT_TO")
    if not (sid and token and from_num and to_num):
        st.sidebar.info("Twilio environment variables not set — SMS will not be sent. Set TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_TO in your app settings.")
        return False
    try:
        client = Client(sid, token)
        msg = client.messages.create(body=body, from_=from_num, to=to_num)
        st.sidebar.success(f"SMS sent (sid: {msg.sid})")
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to send SMS: {e}")
        return False

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Feathered Guardian (Cloud)", page_icon="🐓", layout="wide")
st.title("Feathered Guardian — Streamlit Cloud Compatible")

st.markdown(
    """
    This app runs YOLO inference on uploaded images or uploaded videos **without using OpenCV**.
    - For images: it'll display annotated image inline.
    - For videos: YOLO will run prediction and save annotated output in `runs/detect/...`. The app will locate and offer the annotated file for download.
    """
)

# Sidebar: model selection and settings
st.sidebar.header("Model & Settings")
uploaded_model = st.sidebar.file_uploader("Upload custom YOLO model (best.pt) (optional)", type=["pt"])
model_path_on_disk = None
if uploaded_model:
    # save model to repo runtime (not persisted to git); Streamlit Cloud storage is ephemeral but fine for runtime.
    model_path_on_disk = save_temp_uploaded_file(uploaded_model)
    st.sidebar.write(f"Custom model saved to runtime: {model_path_on_disk}")

conf_threshold = st.sidebar.slider("Confidence threshold", min_value=0.05, max_value=0.99, value=0.25, step=0.01)
display_labels = st.sidebar.checkbox("Show detected labels in results", value=True)

# Load model (lazy)
with st.spinner("Loading YOLO model..."):
    model = load_model(model_path_on_disk)

# Input type
st.header("Inputs")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Image")
    st.write("Upload an image or use the camera to capture one.")
    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="image_uploader")
    camera_img = st.camera_input("Or take a photo with your camera")

with col2:
    st.subheader("Video (uploaded)")
    st.write("Upload a short mp4 video. The app will run YOLO and save annotated results to `runs/detect/...`.")
    uploaded_video = st.file_uploader("Upload video (mp4)", type=["mp4"], key="video_uploader")

# Action buttons
st.markdown("---")
cols = st.columns([1, 1, 1])
process_image_btn = cols[0].button("Process Image")
process_camera_btn = cols[1].button("Process Camera Image")
process_video_btn = cols[2].button("Process Uploaded Video")

# ---------------------------
# Image processing flow
# ---------------------------

if process_image_btn or process_camera_btn:
    # choose source image
    if process_image_btn and not uploaded_image:
        st.warning("Please upload an image first.")
    else:
        try:
            if process_camera_btn and camera_img:
                image = Image.open(BytesIO(camera_img.getvalue()))
            else:
                image = Image.open(BytesIO(uploaded_image.read()))
            st.image(image, caption="Input Image", use_column_width=True)
            with st.spinner("Running YOLO inference on image..."):
                annotated_np, result = run_inference_on_image(model, image, conf=conf_threshold)
                # convert numpy to PIL
                annotated_pil = Image.fromarray(annotated_np)
            st.subheader("Annotated Image")
            st.image(annotated_pil, use_column_width=True)

            # Show detections summary
            if result.boxes is not None and len(result.boxes) > 0:
                detections = []
                for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                    label = model.model.names[int(cls)] if int(cls) in model.model.names else str(int(cls))
                    detections.append({"label": label, "confidence": float(conf), "box": [float(x) for x in box]})
                st.subheader("Detections")
                st.table(detections)
                # Optionally send SMS alert if no raking detected (example logic)
                # Replace with your own business rule; here we demonstrate:
                # Send SMS if no detections found OR if detections include a class called "rake" (example)
                if len(detections) == 0:
                    st.warning("No objects detected.")
                    # Example SMS (only if env vars set)
                    send_sms = st.sidebar.checkbox("Send SMS if no detections", value=False)
                    if send_sms:
                        send_sms_twilio("Alert: No objects detected in the processed image.")
            else:
                st.info("No detections found.")
        except Exception as e:
            st.error(f"Error during image processing: {e}")

# ---------------------------
# Video processing flow
# ---------------------------

if process_video_btn:
    if not uploaded_video:
        st.warning("Please upload an MP4 video first.")
    else:
        try:
            # Save uploaded video to temp path
            video_path = save_temp_uploaded_file(uploaded_video)
            st.write(f"Saved uploaded video to: `{video_path}`")
            st.info("Running YOLO prediction on the uploaded video. This may take a while depending on model size.")
            start = time.time()
            # Run YOLO predict on file and SAVE annotated outputs (save=True)
            # We set visualize=False to avoid opening GUI windows
            results = model.predict(source=video_path, conf=conf_threshold, save=True, verbose=False)
            duration = time.time() - start
            st.success(f"Inference finished in {duration:.1f} s. Searching for saved annotated outputs...")

            # Attempt to find saved annotated outputs in runs/detect
            run_dir = find_latest_run_detect_dir("runs/detect")
            found_files = []
            if run_dir:
                found_files = find_saved_outputs_in_run(run_dir, uploaded_filename=uploaded_video.name)
            if not found_files:
                st.warning("Could not locate annotated output files in runs/detect. Check server logs or model.predict output.")
            else:
                st.write(f"Found {len(found_files)} file(s) in `{run_dir}`. Newest files shown first.")
                for f in found_files[:10]:
                    st.write(f"- `{f}`")
                # Offer the most likely annotated video for download (first mp4 found containing the uploaded filename stem, else the newest mp4)
                mp4s = [f for f in found_files if f.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]]
                if mp4s:
                    annotated_video_path = mp4s[0]
                    st.video(str(annotated_video_path))
                    with open(annotated_video_path, "rb") as vf:
                        btn = st.download_button("Download annotated video", vf.read(), file_name=annotated_video_path.name, mime="video/mp4")
                else:
                    # no video file — maybe YOLO saved frames instead. Offer a zip (optional)
                    st.info("No annotated video file found; YOLO may have saved annotated frames (images). You can download them individually from the run folder.")
                    # show image thumbnails
                    images = [f for f in found_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                    if images:
                        cols = st.columns(3)
                        for i, imgp in enumerate(images[:9]):
                            try:
                                img = Image.open(imgp)
                                cols[i % 3].image(img, caption=imgp.name, use_column_width=True)
                            except Exception:
                                pass

            # cleanup temp video
            try:
                os.remove(video_path)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Error during video processing: {e}")

# ---------------------------
# Footer / Tips
# ---------------------------

st.markdown("---")
st.markdown(
    """
    **Deployment tips**
    - Add `requirements.txt` to your repo (see below).
    - If you want Twilio SMS, set the environment variables in Streamlit Cloud settings: `TWILIO_SID`, `TWILIO_TOKEN`, `TWILIO_FROM`, `ALERT_TO`.
    - If you have a custom `best.pt`, upload it in the sidebar or commit it to your repo (beware of large model sizes > 100MB).
    """
)
