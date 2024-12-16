import os
import face_recognition
import numpy as np
import cv2

# Path to the dataset directory where the images are stored
dataset_dir = r"C:\Users\Sakshi\Desktop\FaceRecognitionProject\dataset"  # Updated path

# Lists to store known face encodings and names
known_face_encodings = []
known_face_names = []

def load_dataset(dataset_path):
    """Load all images in the dataset and create face encodings."""
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):  # Ensure it's a directory
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:  # Ensure a face encoding exists
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                        print(f"Added encoding for {person_name} from {image_file}")
                    else:
                        print(f"No face detected in {image_file}. Skipping.")
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")

# Load the dataset and create encodings
load_dataset(dataset_dir)
print(f"Total faces loaded: {len(known_face_names)}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to quit the video stream.")

# Start the webcam feed for face recognition
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Process each detected face
    for i, (top, right, bottom, left) in enumerate(face_locations):
        face_encoding = face_encodings[i]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, get the name of the matched person
        if matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale back the coordinates to the original frame size
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        # Draw rectangle around the face and put the name below it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Video", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting video stream.")
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
