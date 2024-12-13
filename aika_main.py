import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import face_recognition
import serial

# Initialize serial communication with Arduino (Adjust COM port and baud rate as needed)
#arduino = serial.Serial(port="COM3", baudrate=9600, timeout=1)  # Replace COM3 with your Arduino's port
#time.sleep(2)  # Allow Arduino to initialize


# Pre-trained YOLOv8 model
model = YOLO("yolov8s.pt")
threat_objects = ["knife", "smoke", "needle", "scissors","chainsaw","gun"]

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load known face encodings and names
known_face_encodings = [
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\python project\face recognition\WIN_20241118_15_00_18_Pro.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\python project\face recognition\Screenshot 2024-11-18 145823.png"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-20 023526.png   "))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-20 023701.png   "))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-20 023910.png   "))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-20 024039.png   "))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-22 125914.png  "))[0],
    face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\hiyad\OneDrive\Pictures\Screenshots\Screenshot 2024-11-27 115416.png  "))[0]
]

known_face_names = ["Hiya", "Sakshi", "Farooque", "Dr. Shahab Saquib Sohail", "Dakshish", "Divit","Dr Praveen Lalwani","Dr. Ashfaq Ahmad Najar"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables for face detection smoothing
smooth_factor = 10  # Adjust this value for more or less smoothing
x_history = []
y_history = []

previous_hand_positions = []
movement_threshold = 50  # Threshold for suspicious movement

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2  # Frame center

    # Process frame for face detection (smoothing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # Use the first detected face
        x, y, w, h = faces[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Add the coordinates to the history
        x_history.append(face_center_x)
        y_history.append(face_center_y)

        # Keep history limited to smooth_factor
        if len(x_history) > smooth_factor:
            x_history.pop(0)
            y_history.pop(0)

        # Calculate smoothed center
        smooth_x = int(sum(x_history) / len(x_history))
        smooth_y = int(sum(y_history) / len(y_history))

        # Calculate the offset
        offset_x = smooth_x - center_x
        offset_y = smooth_y - center_y

        # Display offset and draw smoothing rectangle
        cv2.putText(frame, f"Offset: ({offset_x}, {offset_y})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (smooth_x - 5, smooth_y - 5), 
                      (smooth_x + 5, smooth_y + 5), (255, 0, 0), -1)

    # Draw frame center
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot at center

    # Process frame for hand detection (Mediapipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(frame_rgb)

    current_hand_positions = []
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
           # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist = hand_landmarks.landmark[0]
            h, w, _ = frame.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            current_hand_positions.append((wrist_x, wrist_y))
            
    if previous_hand_positions:
        for i, current_pos in enumerate(current_hand_positions):
            if i < len(previous_hand_positions):
                prev_pos = previous_hand_positions[i]
                movement = np.linalg.norm(np.array(prev_pos) - np.array(current_pos))
                if movement > movement_threshold:
                    cv2.putText(frame, "Suspicious Movement!", (current_pos[0], current_pos[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    previous_hand_positions = current_hand_positions

    # Process frame for object detection (YOLO)
    results = model(frame, verbose=False) #added the verbose part to avoid overprocessing
    for result in results:
      for det in result.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = float(det.conf)  # Confidence as a float
        class_id = int(det.cls[0])
        label = model.names[class_id]
        if conf > 0.3:
            if label in threat_objects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"THREAT: {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif label != "person":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Process frame for facial recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Integrated System: Threat Detection, Hand Tracking, Facial Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

