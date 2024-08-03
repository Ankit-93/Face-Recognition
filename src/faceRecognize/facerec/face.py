import cv2
import pickle
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cosine

from faceRecognize.facerec.architecture import *
from faceRecognize.facerec.train_v2 import normalize,l2_normalizer
from faceRecognize.facerec.mobilenet import *


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
confidence_t=0.99
recognition_t=0.5
required_size = (160,160)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def no_detect(img ,detector,encoder,encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    try:
        for res in results:
            face, pt_1, pt_2 = get_face(img_rgb, res['box'])
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
    except:
        pass
    return img 

def face_detector(image):
    #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(image)

    # Draw the face detections and crop the detected faces
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cropped_face = image[y:y+h, x:x+w]

            return cropped_face, x, y, w, h
    else:
        return "None",0,0,0,0



def detect(img ,detector,encoder,encoding_dict):
    #
    face, x, y, w, h = detector(img)
    if face is not None:
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist
         
        if name == 'unknown':
            #cv2.rectangle(img,(x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, name, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            #cv2.rectangle(img,(x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, name + f'_{1-distance:.2f}',(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
        return img
    else:
        return img




if __name__ == "__main__":
    required_shape = (160,160)
    # face_encoder = InceptionResNetV2()
    # path_m = "facenet_keras_weights.h5"
    # face_encoder.load_weights(path_m)
    face_encoder = build_mobilenetv2(3)
    encodings_path = './faceRecognize/facerec/encodings/encodings.pkl'
    encoding_dict = load_pickle(encodings_path)
    COUNT = 0
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret,frame = cap.read()
        

        if not ret:
            print("CAM NOT OPEND") 
            break
        #frame  = detect(frame , face_detector , face_encoder , encoding_dict)
        print(frame.shape)
        try:
            frame  = detect(frame , face_detector , face_encoder , encoding_dict)
        except:
            pass
        cv2.imshow('camera', frame)
        COUNT+=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
