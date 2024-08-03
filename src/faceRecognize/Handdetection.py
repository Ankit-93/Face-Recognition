#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from pygame import mixer 
import json
import pyaudio
import wave
import sys
import csv
import copy
import argparse
import itertools
import time
import winsound
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
print("imported all the libraries")
from model import PointHistoryClassifier


import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier
from utils import CvFpsCalc


# Load your KeyPointClassifier and other necessary modules here
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
keypoint_classifier = KeyPointClassifier()

cvFpsCalc = CvFpsCalc(buffer_len=10)
point_history = []
keypoint_classifier_labels = {
    0: "Fist",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Rock",
    7: "Spock",
    8: "Live long and prosper"
}

def calc_bounding_rect(frame, landmarks):
    # Calculate bounding box
    brect = cv.boundingRect(np.array([landmark for landmark in landmarks]))
    cv.rectangle(frame, (brect[0], brect[1]), (brect[0] + brect[2], brect[1] + brect[3]), (0, 255, 0), 2)
    return brect

def calc_landmark_list(frame, landmarks):
    # Calculate landmark list
    landmark_list = []
    for i, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        landmark_list.append([x, y])
        cv.circle(frame, (x, y), 3, (0, 0, 255), thickness=5)
    return landmark_list

def pre_process_landmark(landmark_list):
    # Convert to relative coordinates / normalized coordinates
    pre_processed_landmark_list = []
    base_x, base_y = landmark_list[0]
    for landmark in landmark_list[1:]:
        pre_processed_landmark_list.append([(landmark[0] - base_x), (landmark[1] - base_y)])
    return pre_processed_landmark_list

def draw_bounding_rect(use_brect, frame, brect):
    # Draw bounding box
    if use_brect:
        cv.rectangle(frame, (brect[0], brect[1]), (brect[0] + brect[2], brect[1] + brect[3]), (255, 0, 0), 2)
    return frame

def draw_landmarks(frame, landmark_list):
    # Draw landmarks
    for landmark in landmark_list:
        cv.circle(frame, (landmark[0], landmark[1]), 5, (0, 255, 0), thickness=-1)
    return frame

def draw_info_text(frame, brect, handedness, hand_sign):
    # Draw information text
    info_text = f"Handedness: {handedness.classification[0].label}"
    cv.putText(frame, info_text, (brect[0], brect[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    cv.putText(frame, hand_sign, (brect[0], brect[1] - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    return frame

def draw_point(frame, point):
    # Draw point
    cv.circle(frame, (point[0], point[1]), 5, (0, 255, 0), thickness=-1)
    return frame

def draw_line(frame, point1, point2):
    # Draw line
    cv.line(frame, (point1[0], point1[1]), (point2[0], point2[1]), (0, 0, 255), thickness=3)
    return frame

def draw_info(frame, fps, mode_text, keypoint_classifier_labels):
    # Draw additional information
    inf = [
        ("Mode", mode_text),
        ("FPS", f"{fps:.2f}"),
    ]
    for i, (key, value) in enumerate(inf):
        y = 20 + i * 20
        x = 20
        cv.putText(frame, f"{key}: {value}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

    y = 50
    for i, (key, value) in enumerate(keypoint_classifier_labels.items()):
        y = 80 + i * 20
        x = 20
        cv.putText(frame, f"{key}: {value}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    return frame

def HAND(frame):
    global point_history
    global keypoint_classifier

    debug_image = frame

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands.process(frame)
    frame.flags.writeable = True

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            # Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == "Not applicable":
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id])

    else:
        point_history.append([0, 0])

    for i, point in enumerate(point_history):
        if point[0] == 0 and point[1] == 0:
            continue
        debug_image = draw_point(debug_image, point)
        if i != 0 and point_history[i - 1][0] != 0 and point_history[i - 1][1] != 0:
            debug_image = draw_line(debug_image, point_history[i - 1], point)

    mode_text = "Mode: Normal"
    debug_image = draw_info(debug_image, 0.0, mode_text, keypoint_classifier_labels)

    return debug_image
