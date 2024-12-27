import cv2
import torch
import mediapipe as mp
import numpy as np
import time
import warnings

# Suppress Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize MediaPipe Pose and FaceMesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Sensitivity Variables
SENSITIVITY = 50  # Default sensitivity
MOTION_THRESHOLD = 1000  # Minimum contour area to detect movement

# Facial Expression Map
FACIAL_EXPRESSIONS = {
    "smile": "Smiling 😊",
    "neutral": "Neutral 😐",
    "frown": "Frowning 😟"
}

# Detect facial expressions based on landmarks
def detect_facial_expression(landmarks):
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y

    if lower_lip - upper_lip > 0.02:
        return "smile"
    elif lower_lip - upper_lip < 0.01:
        return "frown"
    else:
        return "neutral"

# Camera Selection
def select_camera():
    print("🔍 Scanning for available cameras...")
    available_cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    if not available_cameras:
        print("❌ No cameras detected. Exiting.")
        exit()

    print("✅ Available cameras:")
    for idx in available_cameras:
        print(f"[{idx}] Camera {idx}")

    while True:
        try:
            selected_index = int(input("🎥 Enter the camera index to use: "))
            if selected_index in available_cameras:
                return selected_index
            else:
                print("⚠️ Invalid index. Please select from the available cameras.")
        except ValueError:
            print("⚠️ Please enter a valid number.")


# Start Camera
camera_index = select_camera()
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"❌ Failed to open camera at index {camera_index}")
    exit()


print("✅ Motion Detection with AI started. Use '+' to increase sensitivity, '-' to decrease, and 'q' to quit.")

previous_frame = None

# Timer for responsiveness
last_key_time = time.time()
last_detection_time = 0

# Cache last detected pose
last_pose_landmarks = None

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Full-color display frame
    display_frame = frame.copy()

    # Motion Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame is None:
        previous_frame = gray
        continue

    frame_diff = cv2.absdiff(previous_frame, gray)
    _, thresh = cv2.threshold(frame_diff, SENSITIVITY, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    detected_objects = []

    for contour in contours:
        if cv2.contourArea(contour) < MOTION_THRESHOLD:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Object Detection (YOLOv5)
    results = model(frame)
    person_detected = False

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        confidence = float(conf)

        if confidence > 0.5:
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(display_frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            detected_objects.append((label, confidence))

            if label == 'person':
                person_detected = True
                results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results_pose.pose_landmarks:
                    last_pose_landmarks = results_pose.pose_landmarks
                if last_pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame, last_pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                face_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if face_results.multi_face_landmarks:
                    expression = detect_facial_expression(face_results.multi_face_landmarks[0].landmark)
                    print(f"🧠 Detected Facial Expression: {FACIAL_EXPRESSIONS[expression]}")

    # Display Sensitivity Level
    cv2.putText(display_frame, f"Sensitivity: {SENSITIVITY}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('AI Motion Detection', display_frame)

    # Update Previous Frame
    previous_frame = gray

    # Immediate Key Responsiveness
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        SENSITIVITY = max(10, SENSITIVITY - 5)
    elif key == ord('-'):
        SENSITIVITY = min(100, SENSITIVITY + 5)

    # Maintain smooth frame rate
    elapsed_time = time.time() - start_time
    if elapsed_time < 0.05:
        time.sleep(0.05 - elapsed_time)

cap.release()
cv2.destroyAllWindows()
