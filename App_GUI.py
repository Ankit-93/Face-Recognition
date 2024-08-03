import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

from threading import Thread
from src.faceRecognize.face import Faces
from src.faceRecognize.facerecognition import *
import time,string,random,shutil
from pydub import AudioSegment
from pydub.playback import play


def generate_unique_string():
    timestamp = str(int(time.time()))  # Current timestamp
    random_part = ''.join(random.choices(string.ascii_letters + string.digits, k=8))  # Random part
    unique_string = timestamp + '_' + random_part
    return unique_string

def load_alert_sound():
    song = AudioSegment.from_mp3('./src/faceRecognize/audio/alert.wav')
    return song
    
def beautify_showinfo(title, message):
    messagebox.showinfo(title, message, icon="info")

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition App")
        background_image = Image.open("./src/faceRecognize/Background_files/Learn-Facial-Recognition-1024x718.jpg")
        self.background_photo = ImageTk.PhotoImage(background_image)
        self.training_result = None
        self.canvas = tk.Canvas(self.master, width=background_image.width, height=background_image.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)
        self.capture_label = tk.Label(self.canvas)
        self.capture_label.place(relx=0.05, rely=0.15, anchor=tk.NW)
        self.input_frame = tk.Frame(self.canvas, bg="white", bd=5, highlightthickness=0)
        self.input_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        transparent_img = tk.PhotoImage(width=background_image.width, height=background_image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=transparent_img)
        self.input_frame.place(in_=self.canvas, relx=0.7, rely=0.5, anchor=tk.W)
        self.loading_animation = None
        self.animation_running = False
        
        self.cap1 = None
        self.cap2 = None

        self.create_widgets()

    def create_widgets(self):

        param_entries = [i for i in range(150)]

        run_button = tk.Button(self.input_frame, text="Run Code", command=self.run_code_thread,bg="white", fg="blue")
        run_button.grid(row=len(param_entries) + 1,padx=10, pady=5,column=0,sticky=tk.W) #column=0, columnspan=2, pady=5

        quit_button = tk.Button(self.input_frame, text="Quit", command=self.quit_app,bg="white", fg="blue")
        quit_button.grid(row=len(param_entries) + 5,padx=10, pady=5,column=0,sticky=tk.W ) #, column=0, columnspan=2, pady=5

        # Create a button to take a picture
        self.take_picture_button = tk.Button(self.canvas, text="Create User/Add Photos", command=self.capture_image, bg="white", fg="blue")
        self.take_picture_button.place(relx=0.05, rely=0.05, anchor=tk.NW)

        self.delete_folder_button = tk.Button(self.canvas, text="Delete User", command=self.delete_folder,bg="white", fg="blue")
        self.delete_folder_button.place(relx=0.05, rely=0.10, anchor=tk.NW)
        self.animation_label = tk.Label(self.master, text="Training in progress...", font=("Arial", 12))
        self.training_button = tk.Button(self.canvas, text="Start Training", command=self.start_training,bg="white", fg="blue")
        self.training_button.place(relx=0.05, rely=0.15, anchor=tk.NW)



    def run_code(self):
        alert_sound = load_alert_sound()
        face_encoder = model_selector("Facenet")
        encodings_path = './src/faceRecognize/facerec/encodings/encodings.pkl'
        encoding_dict = load_pickle(encodings_path)
        cv2.namedWindow("Video Grid")
        COUNT = 0
        while True:
            try:
                self.cap1 = cv2.VideoCapture(0)
                ret1, frame = self.cap1.read()
                frame = cv2.resize(frame, (640, 300))
                frame_ = frame
                frame4 = Faces(frame_)
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
                    top_row = cv2.hconcat([frame, frame])
                    bottom_row = cv2.hconcat([frame4, frame4])
                    grid = cv2.vconcat([top_row, bottom_row])
                cv2.imshow("Video Grid", grid)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print("Exception:", e)
                pass

        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
        cv2.destroyAllWindows()

    def run_code_thread(self):
        Thread(target=self.run_code).start()

    def quit_app(self):
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
        cv2.destroyAllWindows()
        self.master.quit()

    def capture_image(self):
        choice = messagebox.askyesno("Capture Image", "Do you want to create new user?")
        if choice:
            self.create_new_folder()
        else:
            path = self.choose_existing_folder()
            self.click(path)

    def create_new_folder(self):
        initial_folder_path = "./src/faceRecognize/facerec/data"
        new_folder_name = simpledialog.askstring("Create New Folder", "Enter new folder name:")
        if new_folder_name:
            new_folder_path = os.path.join(initial_folder_path, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            messagebox.showinfo("Success", f"Folder '{new_folder_name}' created successfully.")
            self.capture_image()  # Recursively call capture_image to proceed with capturing after creating the folder.

    def choose_existing_folder(self):
        initial_folder_path = "./src/faceRecognize/facerec/data"
        folder_path = filedialog.askdirectory(title="Select Folder", initialdir=initial_folder_path)
        if not folder_path:
            return  # If no folder selected, exit the function
        else:
            messagebox.showinfo("Selected Folder", f"You selected the folder:\n{folder_path}")
            return folder_path
            
    def click(self, path):
        try:
            print("Initializing video capture...")
            cap1 = cv2.VideoCapture(self.videosource1_var.get(), cv2.CAP_DSHOW)

            if not cap1.isOpened():
                print("Failed to open video capture.")
                return

            ret1, frame1 = cap1.read()
            if ret1:
                frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame1_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.capture_label.config(image=imgtk)
                self.capture_label.image = imgtk

                choice = messagebox.askyesno("Capture Image", "Do you want to capture this image?")
                if choice:
                    filename = str(generate_unique_string())+'.jpg'
                    filename = os.path.join(path,filename)
                    filename = filename.replace("\\",'/')
                    if filename:
                        cv2.imwrite(filename, frame1)
                        beautify_showinfo("Info","Picture saved successfully:")
                    choice = messagebox.askyesno("Capture Image", "Do you want to capture more images?")
                    if choice:
                        self.click(path)
                    else:
                        if cap1 is not None:
                            cap1.release()
                        self.master.destroy()
                        FaceRecognitionApp()
                else:
                    self.click(path)
            else:
                print("Failed to capture frame.")
        except Exception as e:
            print("Error:", e)

        if cap1 is not None:
            cap1.release()

    def delete_folder(self):
        initial_folder_path = "./src/faceRecognize/facerec/data"
        folder_path = filedialog.askdirectory(title="Select User to Delete", initialdir=initial_folder_path)
        if folder_path:
            try:
                shutil.rmtree(folder_path)
                messagebox.showinfo("Success", "Folder deleted successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete folder: {str(e)}")

    def start_training(self):
        self.training_button.pack_forget()
        self.start_animation()
        Thread(target=self.training_thread).start()

    def start_animation(self):
        self.animation_label.pack()
        self.update_animation()

    def stop_animation(self):
        # Hide the animation label
        self.animation_label.pack_forget()

    def training_thread(self):
        # Simulate training process
        time.sleep(5)
        import train_v2
        self.training_result = "Training completed successfully."
        self.stop_animation()
        self.show_training_result()

    def update_animation(self):
        if self.training_result is None:
            self.master.after(100, self.update_animation)
        else:
            self.stop_animation()

    def show_training_result(self):
        messagebox.showinfo("Training Result", self.training_result)


def face_recognition_main():
    root = tk.Tk()
    FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    face_recognition_main()

# FaceRecognitionApp().run_code()
