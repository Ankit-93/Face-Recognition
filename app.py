import warnings
warnings.filterwarnings("ignore")
import av, os, sys, cv2
import streamlit as st
from pydub.playback import play
import time, string, random, shutil
from PIL import Image
import streamlit as st
from pydub import AudioSegment
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration

from src.faceRecognize.face import Faces
from src.faceRecognize.facerecognition import *

sys.path.append(os.path.abspath('src/faceRecognize'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def generate_unique_filename():
    chars = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(chars, k=8))
    return random_string + '.jpg'

def load_alert_sound():
    song = AudioSegment.from_file('src/faceRecognize/audio/alert.wav', format="wav")
    return song

def create_new_folder():
    initial_folder_path = "src/faceRecognize/facerec/data"
    new_folder_name = st.text_input("Enter New User:")
    if new_folder_name and st.button("Create User"):
        new_folder_path = os.path.join(initial_folder_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        st.success(f"User '{new_folder_name}' created successfully.")
        return new_folder_path


def delete_folder():
    base_dir = os.path.join(os.getcwd(),"src/faceRecognize/facerec/data")
    folder_options = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    folder_dict = {}
    for folder in folder_options:
        key = folder.split("/")[-1]
        folder_dict[key] = folder
    selected_folder = st.selectbox("Select User to Delete", list(folder_dict.keys()))
    if selected_folder and st.button("Delete User"):
        try:
            shutil.rmtree(folder_dict[selected_folder])
            st.success("User deleted successfully.")
        except Exception as e:
            st.error(f"Failed to delete user: {str(e)}")

def start_training():
    st.write("Training in progress...")
    time.sleep(5)
    import src.faceRecognize.facerec.train_v2
    st.success("Training completed successfully.")



class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.alert_sound = load_alert_sound()
        self.face_encoder = model_selector("Facenet")
        self.encodings_path = './src/faceRecognize/facerec/encodings/encodings.pkl'
        self.encoding_dict = load_pickle(self.encodings_path)
        self.COUNT = 0

    def recv(self, frame):
        print("Cslling")
        img = frame.to_ndarray(format="bgr24")
        frame_resized = cv2.resize(img, (320, 240))
        frame4 = Faces(frame_resized)
        try:
            frame2, pred = detect(frame_resized, face_detector, self.face_encoder, self.encoding_dict)
            if pred == 'unknown':
                if self.COUNT < 10:
                    self.COUNT += 1
                else:
                    play(self.alert_sound)
            else:
                self.COUNT = 0
            
            top_row = cv2.hconcat([frame2, frame2])
            bottom_row = cv2.hconcat([frame4, frame4])
            grid = cv2.vconcat([top_row, bottom_row])
        except Exception as e:
            top_row = cv2.hconcat([img, img])
            bottom_row = cv2.hconcat([frame4, frame4])
            grid = cv2.vconcat([top_row, bottom_row])
        
        return av.VideoFrame.from_ndarray(grid, format="bgr24")

class YourVideoProcessorClass:
    def recv(self, frame):
        # Process the frame
        return frame

def main():
    st.title("Face Recognition App")
    option = st.sidebar.selectbox("Choose an option", 
                        ("Add User", "Delete User", "Start Training","Run Code"))

    if option == "Add User":
        create_new_folder()

    elif option == "Delete User":
        delete_folder()

    elif option == "Start Training":
        start_training()

    elif option == "Run Code":
        RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]}
            ]})

        webrtc_ctx = webrtc_streamer(
            key="face-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YourVideoProcessorClass,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=FaceRecognitionProcessor().recv,
            media_stream_constraints={"video": True, "audio": False}
            #async_processing=True
        )
        
if __name__ == '__main__':
    main()
# streamlit run app.py