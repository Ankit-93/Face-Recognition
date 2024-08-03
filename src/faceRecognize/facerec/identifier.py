# import cv2
# import mediapipe as mp 

# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
# # Initialize the MTCNN detector

# # Create a VideoCapture object
# cap = cv2.VideoCapture(0)

# with mp_face_detection.FaceDetection(model_selection = 0 , min_detection_confidence = 0.5) as face_detection:
#     while cap.isOpened():
#         ret,image = cap.read()
#         image.flags.writeable = False
#         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#         results = face_detection.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         if results.detections:
#             for face_no,face in enumerate(results.detections):
#                 mp_drawing.draw_detection(image = image,detection = face)
#         cv2.imshow('Face Detection', cv2.flip(image,1))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the capture and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(rgb_image)

    # Draw the face detections
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

    # Display the output
    cv2.imshow('Face Detection', image)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


