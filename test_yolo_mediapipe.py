import torch
import mediapipe as mp

# Verify YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print("✅ YOLOv5 model loaded successfully.")

# Verify MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
print("✅ MediaPipe Pose loaded successfully.")
