import cv2
import mediapipe as mp
import math
from pydub import AudioSegment
from pydub.playback import play

def process_frame(frame, point_1, point_2, point_3, point_4, distance_threshold,hand_close_count_1,hand_close_count_2,hand_close_count_4):
    # Play an alert sound
    sound_path = 'alert_sound.wav'  # Path to the alert sound file
    alert_sound = AudioSegment.from_file(sound_path)

    # Colors
    POINT_1_COLOR = (0, 255, 0)  # Green color for point 1
    POINT_2_COLOR = (0, 0, 255)  # Red color for point 2
    POINT_3_COLOR = (255, 0, 0)  # Blue color for point 3
    POINT_4_COLOR = (255, 255, 0)

    # Variables
    hand_close_to_point_1 = False
    hand_close_to_point_2 = False
    hand_close_to_point_3 = False
    hand_close_to_point_4 = False

    # Mediapipe initialization
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    with mp_hands.Hands(min_detection_confidence=0.25, min_tracking_confidence=0.25) as hands:
        results = hands.process(frame_rgb)

    # Reset flags for each frame
    hand_close_to_point_1 = False
    hand_close_to_point_2 = False
    hand_close_to_point_3 = False
    hand_close_to_point_4 = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand landmarks
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Get landmark position
                cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                # Calculate distance to point 1
                distance_to_point_1 = math.sqrt((cx - point_1[0]) ** 2 + (cy - point_1[1]) ** 2)

                # Calculate distance to point 2
                distance_to_point_2 = math.sqrt((cx - point_2[0]) ** 2 + (cy - point_2[1]) ** 2)

                # Calculate distance to point 3
                distance_to_point_3 = math.sqrt((cx - point_3[0]) ** 2 + (cy - point_3[1]) ** 2)

                # Calculate distance to point 4
                distance_to_point_4 = math.sqrt((cx - point_4[0]) ** 2 + (cy - point_4[1]) ** 2)

                # Check if hand is close to point 1
                if distance_to_point_1 < distance_threshold:
                    hand_close_to_point_1 = True

                # Check if hand is close to point 2
                if distance_to_point_2 < distance_threshold:
                    hand_close_to_point_2 = True

                # Check if hand is close to point 3
                if distance_to_point_3 < distance_threshold:
                    hand_close_to_point_3 = True

                if distance_to_point_4 < distance_threshold:
                    hand_close_to_point_4 = True

                # Draw landmark on frame
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                flag = "HH"

    else:
        flag = "NH"

    # Draw points on frame
    cv2.circle(frame, point_1, 5, POINT_1_COLOR, -1)
    cv2.circle(frame, point_2, 5, POINT_2_COLOR, -1)
    cv2.circle(frame, point_3, 5, POINT_3_COLOR, -1)
    cv2.circle(frame, point_4, 5, POINT_4_COLOR, -1)

    # Count hand proximity to points
    if hand_close_to_point_3:
        hand_close_count_1 = 0
        hand_close_count_2 = 0
        hand_close_count_4 = 0
    elif hand_close_count_4 > 0 and hand_close_to_point_1:
        hand_close_count_1 += 1
    elif hand_close_to_point_2:
        hand_close_count_2 += 1
        if hand_close_count_1 < 1:
            play(alert_sound)
    elif hand_close_to_point_4:
        hand_close_count_4 += 1

    # Display hand proximity count on frame
    cv2.putText(frame, f"Point 1: {hand_close_count_1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, POINT_1_COLOR, 2)
    cv2.putText(frame, f"Point 2: {hand_close_count_2}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, POINT_2_COLOR, 2)
    cv2.putText(frame, f"Point 4: {hand_close_count_4}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, POINT_4_COLOR, 2)

    return frame, hand_close_count_1, hand_close_count_2, hand_close_count_4,flag



# def main():
#     hand_close_count_1 = 0
#     hand_close_count_2 = 0
#     hand_close_count_4 = 0
    
#     cap = cv2.VideoCapture(0)
    
#     x1, y1 = 151, 219
#     x2, y2 = 292, 347
#     x3, y3 = 401, 341
#     x4, y4 = 253, 221
#     distance_threshold = 25

#     point1 = (x1, y1)
#     point2 = (x2, y2)
#     point3 = (x3, y3)
#     point4 = (x4, y4)

#     while cap.isOpened():
        
#         ret, frame = cap.read()
        
#         if not ret:
#             continue
        
#         # Call the process_frame function to process the frame
#         processed_frame, hand_close_count_1, hand_close_count_2, hand_close_count_4 = process_frame(frame,point1,point2, point3, point4, distance_threshold,hand_close_count_1, hand_close_count_2, hand_close_count_4)
        
#         cv2.imshow("Frame", processed_frame)
        
#         if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
