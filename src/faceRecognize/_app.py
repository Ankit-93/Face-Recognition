import warnings

warnings.filterwarnings("ignore")
import os
import sys

# Add the faceRecognize module path to the system path
sys.path.append(os.path.abspath('./faceRecognize'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play

from face import Faces
from facerecognition import *
import time
import string
import random
import shutil, uuid


# Helper functions
# def generate_unique_string():
#     timestamp = str(int(time.time()))  # Current timestamp
#     random_part = ''.join(random.choices(string.ascii_letters + string.digits, k=8))  # Random part
#     unique_string = timestamp + '_' + random_part
#     return unique_string

def generate_unique_string():
    random_uuid = uuid.uuid4()
    print("Random UUID:", random_uuid)
    uuid_string = str(random_uuid)
    return uuid_string

def load_alert_sound():
    song = AudioSegment.from_mp3('./faceRecognize/audio/alert.wav')
    return song


# Streamlit Interface
st.title("Face Recognition App")

# Sidebar for options
option = st.sidebar.selectbox("Choose an option",
                              ("Run Code", "Create User/Add Photos", "Delete User", "Start Training"))


# Functionality to run face recognition
def run_code():
    alert_sound = load_alert_sound()
    face_encoder = model_selctor("Facenet")
    encodings_path = './faceRecognize/facerec/encodings/encodings.pkl'
    encoding_dict = load_pickle(encodings_path)
    cv2.namedWindow("Video Grid")
    COUNT = 0
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        frame = cv2.resize(frame, (640, 300))
        frame4 = Faces(frame)
        try:
            frame2, pred = detect(frame, face_detector, face_encoder, encoding_dict)
            if pred == 'unknown':
                if COUNT < 10:
                    COUNT += 1
                else:
                    play(alert_sound)
            else:
                COUNT = 0
            top_row = cv2.hconcat([frame2, frame2])
            bottom_row = cv2.hconcat([frame4, frame4])
            grid = cv2.vconcat([top_row, bottom_row])
        except Exception as e:
            top_row = cv2.hconcat([frame2, frame])
            bottom_row = cv2.hconcat([frame4, frame4])
            grid = cv2.vconcat([top_row, bottom_row])

        st.image(grid, channels="BGR")
        if st.button('Stop'):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_new_folder():
    initial_folder_path = "./faceRecognize/facerec/data"
    new_folder_name = st.text_input("Enter new folder name:")
    if new_folder_name and st.button("Create Folder"):
        new_folder_path = os.path.join(initial_folder_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        st.success(f"Folder '{new_folder_name}' created successfully.")
        return new_folder_path


def choose_existing_folder():
    initial_folder_path = "./faceRecognize/facerec/data"
    folder_path = st.text_input("Choose from existng user:", initial_folder_path)
    if folder_path and st.button("Select Folder"):
        st.success(f"You selected the folder: {folder_path}")
        return folder_path


def capture_image(path):
    print("Capture Image",path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Failed to open video capture.")
        return

    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        st.image(img, caption="Captured Image", use_column_width=True)
        print("Filename", "Save Image")
        if st.button("Save Image"):
            filename = str(generate_unique_string()) + '.jpg'
            print("Filename",filename)
            filename = os.path.join(path, filename)
            print("Filename", filename)
            filename = filename.replace("\\", '/')
            print("Filename",filename)
            if filename:
                cv2.imwrite(filename, frame)
                st.success("Picture saved successfully")
            if st.button("Capture More"):
                capture_image(path)

    cap.release()


def delete_folder():
    initial_folder_path = "./faceRecognize/facerec/data"
    folder_path = st.text_input("Enter folder path to delete:", initial_folder_path)
    if folder_path and st.button("Delete Folder"):
        try:
            shutil.rmtree(folder_path)
            st.success("Folder deleted successfully.")
        except Exception as e:
            st.error(f"Failed to delete folder: {str(e)}")


def start_training():
    st.write("Training in progress...")
    time.sleep(5)
    import train_v2
    st.success("Training completed successfully.")

#streamlit run faceRecognize/_app.py


if option == "Create User/Add Photos":
    sub_option = st.sidebar.selectbox("Choose an option", ("Choose From Existing", "Add User"))
    if sub_option == "Choose From Existing":
        print("Choose From Existing")
        path = choose_existing_folder()
    else:
        print("Add User")
        path = create_new_folder()
    if path:
        print("Got Path",path)
        capture_image(path)
elif option == "Delete User":
    delete_folder()
elif option == "Start Training":
    start_training()
elif option == "Run Code":
    run_code()
