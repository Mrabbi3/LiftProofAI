"""
Full Shoplifting Detection Test
"""

from modules.detection_engine import ShopliftingDetector
import cv2

print("🚀 LiftProof.Ai - Full Shoplifting Detection")
print("="*60)
print("Instructions:")
print("  - Stand in front of camera")
print("  - Put your hand in your pocket and HOLD for 3+ seconds")
print("  - Box will turn YELLOW (suspicious) then RED (alert!)")
print("  - Press 'q' to quit")
print("="*60 + "\n")

# Initialize detector with behavioral analysis
config = {
    'POCKET_DISTANCE_THRESHOLD': 0.15,
    'POCKET_TIME_THRESHOLD': 45,  # ~3 seconds
    'CONFIDENCE_THRESHOLD': 0.5
}

detector = ShopliftingDetector('yolov8n-pose.pt', config)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam!")
    exit()

print("✅ System active! Monitoring for shoplifting behavior...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process with behavioral analysis
    annotated_frame, alerts = detector.process_frame(frame)
    
    # Handle alerts
    if alerts:
        for alert in alerts:
            print(f"🚨 ALERT: {alert['suspicion_type']} detected!")
    
    # Display
    cv2.imshow('LiftProof.Ai - Shoplifting Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Show stats
stats = detector.get_stats()
print("\n" + "="*60)
print("Session Complete")
print("="*60)
print(f"Frames Processed: {stats['frames_processed']}")
print(f"Total Alerts: {stats['total_alerts']}")
print("="*60)
