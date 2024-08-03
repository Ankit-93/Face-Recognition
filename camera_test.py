import threading
import cv2
import streamlit as st
from matplotlib import pyplot as plt

from streamlit_webrtc import webrtc_streamer,WebRtcMode

lock = threading.Lock()
img_container = {"img": None}
from src.faceRecognize.face import Faces
from src.faceRecognize.facerecognition import *
face_encoder = model_selector("Facenet")
encodings_path = './src/faceRecognize/facerec/encodings/encodings.pkl'
encoding_dict = load_pickle(encodings_path)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    frame2, pred = detect(img, face_detector, face_encoder, encoding_dict)
    with lock:
        img_container["img"] = img
    print(pred)

    return frame2


def video_frame_callback(frame: av.VideoFrame):
    return av.VideoFrame.from_ndarray(image, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]}
            ]})

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)