import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import face_recognition
import serial
import time

# Initialize Arduino communication
arduino = serial.Serial('COM9', 9600)  # Update 'COM9' to your Arduino port
time.sleep(2)  # Allow time for the connection to establish

# Pre-trained YOLOv8 model
model = YOLO("yolov8s.pt")
threat_objects = ["knife", "smoke", "needle", "scissors", "chainsaw", "gun"]

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Load known face encodings and names
known_face_encodings = [
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\python project\face recognition\WIN_20241118_15_00_18_Pro.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\python project\face recognition\Screenshot 2024-11-18 145823.png"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-20 023526.png"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-20 023701.png"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-20 023910.png"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-20 024039.png"))[0]
]

known_face_names = ["Hiya", "Sakshi", "Farooque", "Dr. Shahab Saquib Sohail", "Dakshish", "Divit"]

# Initialize webcam
cap = cv2.VideoCapture(1)

# Check if webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Initialize variables for face detection smoothing
smooth_factor = 10
x_history = []

# Center of the frame
ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture frame")
    cap.release()
    exit()

height, width = frame.shape[:2]
center_x = width // 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Face tracking with servo control
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_center_x = x + w // 2  # Horizontal center of the face box

        # Calculate the horizontal offset from the center of the frame
        offset_x = face_center_x - center_x

        # Map offset to servo position (horizontal movement)
        servo_position_x = 90 + (offset_x // 10)  # Adjust this to match your servo's range
        servo_position_x = max(0, min(180, servo_position_x))  # Constrain servo range

        # Send horizontal servo position to Arduino
        arduino.write(f"{servo_position_x}\n".encode())

        # Draw smoothed marker
        cv2.circle(frame, (face_center_x, y + h // 2), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"Servo X Pos: {servo_position_x}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Process frame for hand detection (Mediapipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(frame_rgb)
    mp_drawing = mp.solutions.drawing_utils
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Process frame for object detection (YOLO)
    results = model(frame, verbose=False)
    for result in results:
        for det in result.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf)  # Confidence
            class_id = int(det.cls[0])
            label = model.names[class_id]

            if conf > 0.3:
                color = (0, 255, 0) if label not in threat_objects else (0, 0, 255)
                text = label if label not in threat_objects else f"THREAT: {label}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Process frame for facial recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Only draw the rectangle for face recognition, not duplicate detection
        cv2.putText(frame, name, (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Integrated System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Correctly placed inside the loop

cap.release()
cv2.destroyAllWindows()
arduino.close()

