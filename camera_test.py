import cv2

# List available camera indices
def list_cameras(max_index=5):
    available_cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


# Display available cameras
print("🔍 Scanning for available cameras...")
cameras = list_cameras()
if not cameras:
    print("❌ No cameras detected. Exiting.")
    exit()

print("✅ Available cameras:")
for idx in cameras:
    print(f"[{idx}] Camera {idx}")

# User selects the camera
while True:
    try:
        selected_index = int(input("🎥 Enter the camera index to use: "))
        if selected_index in cameras:
            break
        else:
            print("⚠️ Invalid index. Please select from the available cameras.")
    except ValueError:
        print("⚠️ Please enter a valid number.")

# Open the selected camera
cap = cv2.VideoCapture(selected_index)
if not cap.isOpened():
    print(f"❌ Failed to open camera at index {selected_index}")
    exit()

print(f"✅ Using camera at index {selected_index}")

# Display the camera feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow(f'Camera {selected_index} Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
