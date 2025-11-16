import streamlit as st
import cv2
import numpy as np
import time
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
from io import BytesIO
import os
from ultralytics import YOLO
from twilio.rest import Client






# Add debug logs
def debug_log(message):
    st.sidebar.text(message)




# Function to process webcam input
def process_webcam(model, names, confidence_threshold, window_name):
    debug_log("Starting webcam processing...")
    track_history = defaultdict(lambda: [])
    consecutive_tracking_time = 0
    raking_done = False  # Flag to indicate if raking was done

    cap = cv2.VideoCapture(0)  # Use default webcam (index 0)

    # Calculate the frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_time = time.time()  # Record start time

    output_video = []

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:

                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Annotator Init
                annotator = Annotator(frame, line_width=2)

                for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):
                    if conf >= confidence_threshold:  # Check confidence threshold
                        # Calculate center coordinates of the bounding box
                        center_x = int((box[0] + box[2]) / 2)
                        center_y = int((box[1] + box[3]) / 2)

                        # Draw center dot
                        cv2.circle(frame, (center_x, center_y), 3, colors(int(cls), True), -1)

                        # Annotate object with bounding box
                        annotator.box_label(box, color=colors(int(cls), True), label=f"{names[int(cls)]} ({conf:.2f})")

                        # Store tracking history
                        track = track_history[track_id]
                        track.append((center_x, center_y))
                        if len(track) > 30:
                            track.pop(0)

                        # Plot tracks
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                        cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

                        # Check if any object has been tracked for more than 8 seconds
                        consecutive_tracking_time = time.time() - start_time
                        if consecutive_tracking_time > 8:
                            raking_done = True  # Set flag to indicate raking was done
                            break  # Stop processing webcam feed and return output

            # Append annotated frame to output list
            output_video.append(frame)
            # Display video frame with annotations
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or raking_done:
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return output_video, raking_done

# Function to process uploaded video input
def process_uploaded_video(uploaded_file, model, names, confidence_threshold, window_name):
    debug_log("Starting uploaded video processing...")
    track_history = defaultdict(lambda: [])
    consecutive_tracking_time = 0
    raking_done = False  # Flag to indicate if raking was done

    # Save uploaded file to disk
    temp_file_path = "temp_video.mp4"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file_path)
    assert cap.isOpened(), "Error reading video file"

    # Calculate the frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_time = time.time()  # Record start time

    output_video = []

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:

                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Annotator Init
                annotator = Annotator(frame, line_width=2)

                for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):
                    if conf >= confidence_threshold:  # Check confidence threshold
                        # Calculate center coordinates of the bounding box
                        center_x = int((box[0] + box[2]) / 2)
                        center_y = int((box[1] + box[3]) / 2)

                        # Draw center dot
                        cv2.circle(frame, (center_x, center_y), 3, colors(int(cls), True), -1)

                        # Annotate object with bounding box
                        annotator.box_label(box, color=colors(int(cls), True), label=f"{names[int(cls)]} ({conf:.2f})")

                        # Store tracking history
                        track = track_history[track_id]
                        track.append((center_x, center_y))
                        if len(track) > 30:
                            track.pop(0)

                        # Plot tracks
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                        cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

                        # Check if any object has been tracked for more than 8 seconds
                        consecutive_tracking_time = time.time() - start_time
                        if consecutive_tracking_time > 8:
                            raking_done = True  # Set flag to indicate raking was done
                            break  # Stop processing uploaded video and return output

            # Append annotated frame to output list
            output_video.append(frame)
            # Display video frame with annotations
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or raking_done:
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Remove temporary file
    os.remove(temp_file_path)

    return output_video, raking_done
# Function to send SMS using Twilio
def send_sms(body):
    account_sid = ''
    auth_token = ''
    twilio_phone_number = '+'  # Your Twilio phone number
    recipient_phone_number = '+'  # Recipient's phone number

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=body,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )

    return message.sid

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Feathered Guardian: A Smart Poultry Litter Tracking and Alert System",
        page_icon="🐓",
        layout="wide"
    )

    st.title("Feathered Guardian: A Smart Poultry Litter Tracking and Alert System")

    st.markdown(
        """
        This application allows you to track objects using your webcam or by uploading a video file.
        """
    )

    input_type = st.radio("Select Input Type", ("Uploaded Video", "Webcam"))
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

    st.markdown(
        """
        Adjust the confidence threshold to filter out detections with lower confidence scores.
        """
    )

    if input_type == "Uploaded Video":
        uploaded_file = st.file_uploader("Upload Video File", type=["mp4"])

        if uploaded_file is not None:
            model = YOLO(r"C:\1. Micro storage\360DIGI\Project\Litter Racking Detection and Alert System\Submitted Files\best.pt")
            names = model.model.names

            if st.button("Process Uploaded Video"):
                start_time = time.time()
                with st.spinner("Processing video..."):
                    annotated_video, raking_done = process_uploaded_video(uploaded_file, model, names, confidence_threshold, window_name='Object Tracking')
                end_time = time.time()
                st.write(f"Processing time: {end_time - start_time:.2f} seconds")

                st.markdown("---")
                st.subheader("Download Annotated Video")
                st.download_button(label="Download Annotated Video", data=encode_video(annotated_video), file_name="annotated_video.mp4", mime="video/mp4")

                if not raking_done:
                    st.warning("Raking was not completed.")
                    # Send SMS notification
                    send_sms("Alert! Raking was not completed.")

    elif input_type == "Webcam":
        st.markdown("Click the button below to start object tracking using your webcam.")
        if st.button("Start Webcam"):
            model = YOLO(r"C:\1. Micro storage\360DIGI\Project\Litter Racking Detection and Alert System\Submitted Files\best.pt")
            names = model.model.names

            st.write("Webcam is running...")
            annotated_video, raking_done = process_webcam(model, names, confidence_threshold, window_name='Object Tracking')

            st.write(f"Total time taken: {len(annotated_video) / 30:.2f} seconds")

            st.markdown("---")
            st.subheader("Download Annotated Video")
            st.download_button(label="Download Annotated Video", data=encode_video(annotated_video), file_name="annotated_video.mp4", mime="video/mp4")

            if not raking_done:
                st.warning("Raking was not completed.")
                # Send SMS notification
                send_sms("Alert! Raking was not completed.")

# Function to encode frames into video buffer (Unchanged)
def encode_video(frames):
    output_video = cv2.VideoWriter_fourcc(*'mp4v')
    output_buffer = BytesIO()
    temp_file_path = "temp_annotated_video.mp4"
    out = cv2.VideoWriter(temp_file_path, output_video, 30, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
    with open(temp_file_path, "rb") as f:
        output_buffer.write(f.read())
    os.remove(temp_file_path)
    return output_buffer.getvalue()

if __name__ == "__main__":
    main()