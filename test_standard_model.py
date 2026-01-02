"""
Test with Standard YOLOv8-Pose Model
"""

from ultralytics import YOLO
import cv2

print("🚀 Testing with Standard YOLOv8-Pose Model")
print("="*60)

# Download and load standard model automatically
print("Loading standard model (will download if needed)...")
model = YOLO('yolov8n-pose.pt')
print("✅ Model loaded!")

# Open webcam
print("\nOpening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam!")
    exit()

print("✅ Webcam ready!")
print("\n🎬 Press 'q' to quit")
print("="*60 + "\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Run detection
    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()
    
    # Add info
    cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if results[0].boxes is not None:
        num = len(results[0].boxes)
        cv2.putText(annotated, f"People: {num}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display
    cv2.imshow('Standard YOLOv8-Pose Test', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Test complete! Processed {frame_count} frames")
