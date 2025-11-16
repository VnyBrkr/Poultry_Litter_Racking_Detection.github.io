import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics_lite import YOLO
from twilio.rest import Client

# Load ONNX model
MODEL_PATH = "best.onnx"
model = YOLO(MODEL_PATH)

# Twilio keys (set these in Streamlit Secrets)
account_sid = st.secrets["twilio"]["ACCOUNT_SID"]
auth_token = st.secrets["twilio"]["AUTH_TOKEN"]
client = Client(account_sid, auth_token)
TO_PHONE = st.secrets["twilio"]["TO_NUMBER"]
FROM_PHONE = st.secrets["twilio"]["FROM_NUMBER"]


def send_alert():
    client.messages.create(
        body="⚠ Detection Alert!",
        from_=FROM_PHONE,
        to=TO_PHONE
    )


def draw_boxes(image: Image.Image, detections):
    """Draw bounding boxes on the image (ultralytics-lite format)."""
    draw = ImageDraw.Draw(image)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["class_name"]
        score = det["score"]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Draw label
        text = f"{label} {score:.2f}"
        draw.text((x1, y1 - 10), text, fill="red")

    return image


st.title("YOLO ONNX Detection — Streamlit Cloud Compatible")
st.write("Upload an image or use your camera to run detection.")

option = st.radio("Choose input:", ["Upload Image", "Camera Input"])

img = None

if option == "Upload Image":
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)

elif option == "Camera Input":
    capture = st.camera_input("Take a photo")
    if capture:
        img = Image.open(capture)

if img:
    st.image(img, caption="Input Image", use_container_width=True)

    # Convert to array for model
    img_np = np.array(img)

    # Run inference (ultralytics-lite returns dict list)
    detections = model(img_np)

    # Draw detections
    output_img = draw_boxes(img.copy(), detections)

    st.image(output_img, caption="Detections", use_container_width=True)

    # If detections exist → Send SMS
    if len(detections) > 0:
        st.success(f"{len(detections)} objects detected! Sending SMS alert...")
        send_alert()
    else:
        st.info("No objects detected.")
