import os
import streamlit as st
import cv2
import string
import random
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# Function to generate a unique filename
def generate_unique_filename():
    chars = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(chars, k=8))
    return random_string + '.jpg'

# Custom video processor class
class ImageCaptureProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

    def get_frame(self):
        return self.frame

# Function to capture and save the image
def capture_and_save_image(webrtc_ctx, path):
    if webrtc_ctx.video_processor:
        frame = webrtc_ctx.video_processor.get_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((320, 240))
            st.image(img, caption="Captured Image", use_column_width=True)
            if not st.button("Save Image"):
                filename = generate_unique_filename()
                filepath = os.path.join(path, filename)
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(filepath, frame)
                path = path.split("\\")[-1]
                st.success(f"Saved image for : {path}")

def main():
    # RTC configuration for WebRTC
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    st.title("Image Capture App")
    base_dir = os.path.join(os.getcwd(),"src/faceRecognize/facerec/data")
    folder_options = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    folder_dict = {}
    for folder in folder_options:
        key = folder.split("/")[-1]
        folder_dict[key] = folder
    selected_folder = st.selectbox("Select User to Save Image", list(folder_dict.keys()))

    # Initialize the WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=ImageCaptureProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

    # Capture and save image on button click
    if st.button("Capture Image"):
        capture_and_save_image(webrtc_ctx, folder_dict[selected_folder])

main()
