import streamlit as st
import numpy as np
from PIL import Image
from ultralytics-lite import YOLO
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


st.title("YOLO ONNX Detection — Streamlit Cloud Compatible")
st.write("Upload an image or take one using your camera.")

option = st.radio("Choose input:", ["Upload Image", "Camera Input"])

img = None

if option == "Upload Image":
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)

elif option == "Camera Input":
    camera_capture = st.camera_input("Take a photo")
    if camera_capture:
        img = Image.open(camera_capture)

if img:
    st.image(img, caption="Input Image", use_container_width=True)

    # Convert image to numpy array
    img_np = np.array(img)

    # Run inference
    results = model.predict(img_np)

    # Render results to PIL image
    result_img = results.draw()  # returns a PIL image

    st.image(result_img, caption="Detections", use_container_width=True)

    # Send SMS if objects detected
    if len(results[0].boxes) > 0:
        st.success("Objects detected! Sending SMS alert...")
        send_alert()
    else:
        st.info("No objects detected.")
