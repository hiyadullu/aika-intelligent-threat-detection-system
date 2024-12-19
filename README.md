# AIKA - Integrated Threat Detection and Recognition System

AIKA is an advanced integrated system that combines multiple computer vision techniques to detect and recognize various objects, faces, and hand movements in real-time. This system is designed to identify potential threats, classify objects, track hand gestures, and recognize faces using pre-trained models and frameworks such as YOLOv8, TensorFlow, Mediapipe, and face_recognition.

## Features
- **Object Detection (YOLOv8)**: Identifies potential threats like knives, guns, and other dangerous objects in real-time using the YOLOv8 model.
- **Hand Tracking (MediaPipe)**: Tracks hand movements to detect suspicious actions.
- **Facial Recognition**: Recognizes predefined faces using the `face_recognition` library.
- **Movement Detection**: Flags suspicious hand movements based on a movement threshold.
- **Serial Communication (Arduino)**: Placeholder for integrating Arduino for real-time alerts (currently commented out).

## Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- OpenCV
- NumPy
- MediaPipe
- YOLOv8 Model (via `ultralytics`)
- `face_recognition`
- PySerial (for Arduino communication)

You can install the required Python packages using:
```bash
pip install opencv-python numpy mediapipe ultralytics face_recognition pyserial

## Setup
**1. YOLOv8 Model**
Download the YOLOv8 pre-trained model. The code uses the "yolov8s.pt" file, which can be obtained from Ultralytics YOLOv8 repository.

**2. Face Recognition Images**
Ensure you have images of the faces that need to be recognized, and update the paths in the known_face_encodings list accordingly. The images should be in .jpg or .png formats.

**3. Arduino Integration (Optional)**
For real-time alerts or actions, connect an Arduino to your system via serial communication. Modify the serial port and baud rate according to your setup. (Currently commented out in the code.)

**4. Webcam**
Make sure your webcam is accessible to OpenCV. The code defaults to using the first available camera (cv2.VideoCapture(0)).

## How to Use
1. Run the Python script.
2.The webcam feed will display in a window showing:
    -Object detection results (threat objects highlighted).
    -Hand movement detection (suspicious movement flagged).
    -Facial recognition with names.
3.Press 'q' to exit the program.

## Code Explanation
YOLOv8 Model: Detects objects in the camera feed and highlights any dangerous items.
MediaPipe Hands: Detects and tracks hand movements, flagging suspicious movements based on a defined threshold.
Face Recognition: Compares detected faces with a predefined list of known faces and labels them accordingly.
Face Detection Smoothing: Uses a history of previous face positions to smooth out detection, reducing jitter.
Serial Communication: Placeholder for sending data to an Arduino, such as triggering alerts.
Example Output
(This is a placeholder image for output representation)

Notes
The movement threshold for suspicious hand movements is adjustable.
The threat object list (threat_objects) can be customized based on the use case.
Ensure that your system has sufficient resources to handle the real-time processing demands of this project, especially when running object detection and hand tracking simultaneously.


