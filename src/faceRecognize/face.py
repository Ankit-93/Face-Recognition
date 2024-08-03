import sys
sys.path.append("C:/ProgramData/Anaconda3/envs/facerecog/Lib/site-packages")
# import required modules
from pydub import AudioSegment
from pydub.playback import play

import cv2
import mediapipe as mp
import numpy as np
import json
#from pygame import mixer
import time
# import os
# os.environ['SDL_AUDIODRIVER'] = "dsp"

#import pygame
#pygame.init()
#pygame.mixer.init()


global z
global time1

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

left, right, up, down = 0, 0, 0, 0
leftState, rightState, upState, downState = 1, 1, 1, 1
v1, v2, v3, v4, v5 = 0, 0, 0, 0, 0  # Initialize v1, v2, v3, v4, and v5

#mixer.init()
#mixer.init('alsa')
def Faces(frame):
    global z
    global time1
    global left, right, up, down, leftState, rightState, upState, downState, v1, v2, v3, v4, v5

    man = 0
    end = 0
    start = 0
    str5 = 'Time in minutes : '

    a = []
    m = 0
    min = 0
    rik = 0
    rik1 = 0
    z1 = 0
    z2 = 0
    preval = 0
    starttime = time.perf_counter()
    time1 = time.perf_counter() - starttime

    success = True
    image = frame

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        time1 = time.perf_counter() - starttime

        if int(man) == 1:
            time1 = time1 - (end - m) + 1

        # print("start :", start)
        # print("z :", man)
        # print("end :", end)
        # print("min", rik)

        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])

                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -10:
                v1 = time.perf_counter()
                if (v1 - v5) > 1:
                    # mixer.music.load('./x.mpeg')
                    # mixer.music.play()
                    song = AudioSegment.from_mp3('./faceRecognize/x.mpeg')
                    play(song)
                    print('play song from face')
                if leftState:
                    left = left + 1
                    leftState = 0
                    rightState = 1

                text = "Looking Left"
            elif y > 10:
                v2 = time.perf_counter()
                if (v2 - v5) > 20:
                    song = AudioSegment.from_mp3('./faceRecognize/x.mpeg')
                    play(song)
                    print('play song from face')
                    # mixer.music.load('./x.mpeg')
                    # mixer.music.play()
                if rightState:
                    leftState = 1
                    upState = 1
                    downState = 1
                    rightState = 0
                    right = right + 1
                text = "Looking Right"
            else:
                v5 = time.perf_counter()
                leftState = 1
                rightState = 1
                upState = 1
                downState = 1
                text = "Forward"

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, "Left: " + str(np.round(left, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "Right: " + str(np.round(right, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "Up: " + str(np.round(up, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "Down: " + str(np.round(down, 2)), (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(image, f'time: {int(time1)} sec', (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    else:
        end = time.perf_counter() - starttime
        man = 1
        m = time1

    return image
