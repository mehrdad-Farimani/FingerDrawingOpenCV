# Finger drawing project by OpenCV. In this project I used mediapipe library to detect hand landmarks, specifically the tip of the index finger.
# This was a project created by Mehrdad H.M. Farimani as a remembrance of childhood finger drawing.
# You can use and share but please reserve the credits. 12 May 2024-Sweden.

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hand model- I run it on my macbook pro M1. some modifications to CPU is required.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize webcam (my webcam was my laptop embedded webcam)
cap = cv2.VideoCapture(0)

# Initial coordinates of index finger tip (inside). Only use index finger for drawing and fold the other fingers.
prev_x, prev_y = -1, -1

# List to store all trajectory points
all_trajectory = []

# List to store current trajectory points
trajectory = []

# Initialize hand state (True: hand is open, False: hand is closed)
hand_open = True

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of index finger tip (landmark ID: 8)
            index_finger_tip = hand_landmarks.landmark[8]
            index_finger_x, index_finger_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

            # Check if index finger tip is not on the face
            if index_finger_y < frame.shape[0] // 2:
                # Draw circle at the tip of index finger
                cv2.circle(frame, (index_finger_x, index_finger_y), 5, (0, 0, 255), -1)

                # Check if hand is open or closed
                landmarks_array = np.array([[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z] for i in range(21)])
                thumb_tip = landmarks_array[4]
                index_tip = landmarks_array[8]
                middle_tip = landmarks_array[12]
                ring_tip = landmarks_array[16]
                little_tip = landmarks_array[20]

                # Calculate distances between fingertips
                thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
                index_middle_dist = np.linalg.norm(index_tip - middle_tip)
                middle_ring_dist = np.linalg.norm(middle_tip - ring_tip)
                ring_little_dist = np.linalg.norm(ring_tip - little_tip)

                # Check if hand is closed (fist)
                if thumb_index_dist < index_middle_dist and index_middle_dist < middle_ring_dist and middle_ring_dist < ring_little_dist:
                    hand_open = False
                else:
                    hand_open = True

                # Draw thicker black trajectory if hand is open and previous coordinates exist
                if hand_open and prev_x != -1 and prev_y != -1:
                    trajectory.append((prev_x, prev_y))
                    for i in range(1, len(trajectory)):
                        cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 0), 2)

                # Update previous coordinates if hand is open
                if hand_open:
                    prev_x, prev_y = index_finger_x, index_finger_y
                    all_trajectory.append((index_finger_x, index_finger_y))
                else:
                    # Reset trajectory if hand is closed
                    prev_x, prev_y = -1, -1
                    trajectory = []
                    all_trajectory = []

    # Erase lines if using only pinky finger (it does not work very well, but good luck!)
    if hand_open and all_trajectory:
        for point in all_trajectory:
            x, y = point
            pinky_tip = landmarks_array[20]
            pinky_x, pinky_y = int(pinky_tip[0] * frame.shape[1]), int(pinky_tip[1] * frame.shape[0])
            dist = np.linalg.norm(np.array([x, y]) - np.array([pinky_x, pinky_y]))
            if dist < 10:  # Increase the threshold for erasing
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)  # Increase the size of the erasing circle

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
