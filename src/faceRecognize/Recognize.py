import cv2
import math
import datetime
import os
import pathlib
import pandas as pd
from pydub import AudioSegment
from pydub.playback import play

# Define global variables
tt = ""
conf = ""
unknown_detected = False
known_detected = False

def process_frame(frame):
    global tt
    global conf
    global unknown_detected
    global known_detected

    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(str(harcascadePath))
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (10, 159, 255), 2)
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if conf < 100:
            aa = df.loc[df['Id'] == Id]['Name'].values
            confstr = "  {0}%".format(round(100 - conf))
            tt = str(Id)+"-"+aa
        else:
            Id = '  Unknown  '
            tt = str(Id)
            confstr = "  {0}%".format(round(100 - conf))

        if (100 - conf) > 0:
            if not known_detected:
                known_detected = True
        else:
            if not unknown_detected:
                unknown_detected = True
                print("Unknown Person Detected")
                
        
        if unknown_detected:
            sound_path = 'alert_sound.wav'  # Replace with the path to your alarm sound file for unknown persons
            alert_sound = AudioSegment.from_file(sound_path)
            print("Playing Unknown Person Alarm")  # Debug statement
            play(alert_sound)


        if known_detected and unknown_detected:
            print('Unknown and Known Persons Alarm!')
            sound_path = 'alert_sound.wav'  # Replace with the path to your alarm sound file for both unknown and known persons
            alert_sound = AudioSegment.from_file(sound_path)
            play(alert_sound)
            known_detected = False
            unknown_detected = False

        tt = str(tt)[2:-2]
        if (100 - conf) > 67:
            tt = tt + " [Pass]"
            cv2.putText(frame, str(tt), (x+5, y-5), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        if (100 - conf) > 67:
            cv2.putText(frame, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
        elif (100 - conf) > 50:
            cv2.putText(frame, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
        else:
            cv2.putText(frame, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

    attendance = attendance.drop_duplicates(subset=['Id'], keep='first')

    return frame
