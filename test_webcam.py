"""
LiftProof.Ai - Advanced Shoplifting Detection Engine v2.0
=========================================================
Multi-behavior analysis for accurate shoplifting detection

Detects:
1. Hand-to-pocket concealment (putting items in pocket)
2. Hand-to-waistband hiding (tucking items in pants)
3. Hand-under-shirt concealment (hiding under clothing)
4. Grabbing motion (quick snatching movements)
5. Nervous behavior (looking around, fidgeting)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
from collections import deque

# Keypoint indices for YOLOv8-Pose
KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}


class PersonTracker:
    """Tracks individual person's behavior over time"""
    
    def __init__(self, track_id):
        self.track_id = track_id
        self.first_seen = time.time()
        self.last_alert_time = 0
        
        # Behavior counters (frames)
        self.hand_near_pocket_frames = 0
        self.hand_near_waist_frames = 0
        self.hand_under_shirt_frames = 0
        self.quick_hand_movement_frames = 0
        self.looking_around_frames = 0
        
        # Position history for movement analysis
        self.left_wrist_history = deque(maxlen=30)
        self.right_wrist_history = deque(maxlen=30)
        self.head_history = deque(maxlen=30)
        
        # State tracking
        self.suspicion_score = 0.0
        self.state = "NORMAL"  # NORMAL, WATCHING, SUSPICIOUS, ALERT
        self.current_behaviors = []
        
        # Alert cooldown
        self.alert_cooldown = 0


class AdvancedShopliftingDetector:
    """
    Advanced behavioral analysis for shoplifting detection
    Uses multiple indicators and temporal patterns
    """
    
    def __init__(self, model_path='yolov8n-pose.pt'):
        print("🚀 Loading Advanced Shoplifting Detection Engine v2.0")
        print("=" * 60)
        
        # Load YOLOv8-Pose model
        self.model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        # Person trackers
        self.persons = {}
        
        # Detection thresholds (calibrated for accuracy)
        self.config = {
            # Distance thresholds (relative to body height)
            'POCKET_THRESHOLD': 0.12,      # Hand near hip/pocket
            'WAIST_THRESHOLD': 0.10,       # Hand at waistband
            'CHEST_THRESHOLD': 0.15,       # Hand near chest (under shirt)
            
            # Time thresholds (frames at ~30fps)
            'POCKET_FRAMES': 25,           # ~0.8 seconds in pocket
            'WAIST_FRAMES': 20,            # ~0.7 seconds at waist
            'CHEST_FRAMES': 25,            # ~0.8 seconds at chest
            'QUICK_MOVE_FRAMES': 10,       # Quick grab detection
            'LOOKING_FRAMES': 45,          # ~1.5 seconds looking around
            
            # Movement thresholds
            'QUICK_MOVE_SPEED': 0.08,      # Relative speed for "quick" movement
            'HEAD_MOVE_THRESHOLD': 0.03,   # Head movement for "looking around"
            
            # Alert thresholds
            'ALERT_THRESHOLD': 70,         # Suspicion score to trigger alert
            'SUSPICIOUS_THRESHOLD': 40,    # Score to mark as suspicious
            'WATCHING_THRESHOLD': 20,      # Score to start watching
            
            # Cooldowns
            'ALERT_COOLDOWN': 90,          # Frames between alerts (~3 seconds)
            
            # Confidence
            'MIN_KEYPOINT_CONF': 0.4,      # Minimum keypoint confidence
        }
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_alerts': 0,
            'behaviors_detected': {
                'pocket_concealment': 0,
                'waist_hiding': 0,
                'shirt_concealment': 0,
                'quick_grab': 0,
                'nervous_behavior': 0
            }
        }
        
        print("✅ Detection engine initialized")
        print("=" * 60)
    
    def process_frame(self, frame):
        """Process a single frame and return annotated frame + alerts"""
        self.stats['frames_processed'] += 1
        alerts = []
        
        # Run pose detection with tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=0.5,
            verbose=False
        )
        
        # Check if we have detections
        if results[0].boxes is None or results[0].boxes.id is None:
            return self._draw_ui(frame, []), alerts
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        if results[0].keypoints is None:
            return self._draw_ui(frame, []), alerts
            
        all_keypoints = results[0].keypoints.data.cpu().numpy()
        
        active_persons = []
        
        # Process each detected person
        for i, track_id in enumerate(track_ids):
            bbox = boxes[i]
            keypoints = all_keypoints[i]
            
            # Get or create person tracker
            if track_id not in self.persons:
                self.persons[track_id] = PersonTracker(track_id)
            
            person = self.persons[track_id]
            active_persons.append(person)
            
            # Reduce alert cooldown
            if person.alert_cooldown > 0:
                person.alert_cooldown -= 1
            
            # Analyze behaviors
            behaviors = self._analyze_all_behaviors(person, keypoints)
            person.current_behaviors = behaviors
            
            # Calculate suspicion score
            self._update_suspicion_score(person, behaviors)
            
            # Update state based on score
            self._update_state(person)
            
            # Check for alerts
            if person.state == "ALERT" and person.alert_cooldown == 0:
                alert = self._generate_alert(person, bbox, frame, behaviors)
                alerts.append(alert)
                person.alert_cooldown = self.config['ALERT_COOLDOWN']
                self.stats['total_alerts'] += 1
            
            # Draw person on frame
            frame = self._draw_person(frame, person, bbox, keypoints)
        
        # Draw UI overlay
        frame = self._draw_ui(frame, active_persons)
        
        return frame, alerts
    
    def _analyze_all_behaviors(self, person, keypoints):
        """Analyze all suspicious behaviors"""
        behaviors = []
        
        # Get key body parts with confidence check
        conf = self.config['MIN_KEYPOINT_CONF']
        
        left_wrist = keypoints[KEYPOINTS['left_wrist']]
        right_wrist = keypoints[KEYPOINTS['right_wrist']]
        left_hip = keypoints[KEYPOINTS['left_hip']]
        right_hip = keypoints[KEYPOINTS['right_hip']]
        left_shoulder = keypoints[KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[KEYPOINTS['right_shoulder']]
        nose = keypoints[KEYPOINTS['nose']]
        
        # Calculate body height for relative measurements
        body_height = self._get_body_height(keypoints)
        if body_height < 50:
            return behaviors
        
        # Store positions for movement analysis
        if left_wrist[2] > conf:
            person.left_wrist_history.append((left_wrist[0], left_wrist[1], time.time()))
        if right_wrist[2] > conf:
            person.right_wrist_history.append((right_wrist[0], right_wrist[1], time.time()))
        if nose[2] > conf:
            person.head_history.append((nose[0], nose[1], time.time()))
        
        # ===== BEHAVIOR 1: Hand near pocket =====
        pocket_detected = False
        if left_wrist[2] > conf and left_hip[2] > conf:
            dist = self._distance(left_wrist, left_hip) / body_height
            if dist < self.config['POCKET_THRESHOLD']:
                pocket_detected = True
        
        if right_wrist[2] > conf and right_hip[2] > conf:
            dist = self._distance(right_wrist, right_hip) / body_height
            if dist < self.config['POCKET_THRESHOLD']:
                pocket_detected = True
        
        if pocket_detected:
            person.hand_near_pocket_frames += 2
            if person.hand_near_pocket_frames >= self.config['POCKET_FRAMES']:
                behaviors.append('POCKET_CONCEALMENT')
                self.stats['behaviors_detected']['pocket_concealment'] += 1
        else:
            person.hand_near_pocket_frames = max(0, person.hand_near_pocket_frames - 1)
        
        # ===== BEHAVIOR 2: Hand at waistband (front) =====
        waist_center = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2 - body_height * 0.05
        )
        
        waist_detected = False
        if left_wrist[2] > conf:
            dist = self._distance(left_wrist, waist_center) / body_height
            if dist < self.config['WAIST_THRESHOLD']:
                waist_detected = True
        
        if right_wrist[2] > conf:
            dist = self._distance(right_wrist, waist_center) / body_height
            if dist < self.config['WAIST_THRESHOLD']:
                waist_detected = True
        
        if waist_detected:
            person.hand_near_waist_frames += 2
            if person.hand_near_waist_frames >= self.config['WAIST_FRAMES']:
                behaviors.append('WAIST_HIDING')
                self.stats['behaviors_detected']['waist_hiding'] += 1
        else:
            person.hand_near_waist_frames = max(0, person.hand_near_waist_frames - 1)
        
        # ===== BEHAVIOR 3: Hand under shirt (chest area) =====
        chest_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2 + body_height * 0.15
        )
        
        chest_detected = False
        if left_wrist[2] > conf:
            dist = self._distance(left_wrist, chest_center) / body_height
            if dist < self.config['CHEST_THRESHOLD']:
                chest_detected = True
        
        if right_wrist[2] > conf:
            dist = self._distance(right_wrist, chest_center) / body_height
            if dist < self.config['CHEST_THRESHOLD']:
                chest_detected = True
        
        if chest_detected:
            person.hand_under_shirt_frames += 2
            if person.hand_under_shirt_frames >= self.config['CHEST_FRAMES']:
                behaviors.append('SHIRT_CONCEALMENT')
                self.stats['behaviors_detected']['shirt_concealment'] += 1
        else:
            person.hand_under_shirt_frames = max(0, person.hand_under_shirt_frames - 1)
        
        # ===== BEHAVIOR 4: Quick grabbing motion =====
        quick_move = self._detect_quick_movement(person, body_height)
        if quick_move:
            person.quick_hand_movement_frames += 3
            if person.quick_hand_movement_frames >= self.config['QUICK_MOVE_FRAMES']:
                behaviors.append('QUICK_GRAB')
                self.stats['behaviors_detected']['quick_grab'] += 1
                person.quick_hand_movement_frames = 0  # Reset after detection
        else:
            person.quick_hand_movement_frames = max(0, person.quick_hand_movement_frames - 1)
        
        # ===== BEHAVIOR 5: Looking around nervously =====
        looking_around = self._detect_looking_around(person, body_height)
        if looking_around:
            person.looking_around_frames += 1
            if person.looking_around_frames >= self.config['LOOKING_FRAMES']:
                behaviors.append('NERVOUS_LOOKING')
                self.stats['behaviors_detected']['nervous_behavior'] += 1
        else:
            person.looking_around_frames = max(0, person.looking_around_frames - 1)
        
        return behaviors
    
    def _detect_quick_movement(self, person, body_height):
        """Detect quick hand movements (grabbing motion)"""
        for history in [person.left_wrist_history, person.right_wrist_history]:
            if len(history) >= 5:
                recent = list(history)[-5:]
                total_dist = 0
                for i in range(len(recent) - 1):
                    dist = math.sqrt(
                        (recent[i+1][0] - recent[i][0])**2 + 
                        (recent[i+1][1] - recent[i][1])**2
                    )
                    total_dist += dist
                
                speed = total_dist / body_height
                if speed > self.config['QUICK_MOVE_SPEED']:
                    return True
        return False
    
    def _detect_looking_around(self, person, body_height):
        """Detect nervous looking around behavior"""
        if len(person.head_history) >= 15:
            recent = list(person.head_history)[-15:]
            
            # Calculate total head movement
            total_movement = 0
            for i in range(len(recent) - 1):
                dist = math.sqrt(
                    (recent[i+1][0] - recent[i][0])**2 + 
                    (recent[i+1][1] - recent[i][1])**2
                )
                total_movement += dist
            
            normalized = total_movement / body_height
            if normalized > self.config['HEAD_MOVE_THRESHOLD']:
                return True
        return False
    
    def _update_suspicion_score(self, person, behaviors):
        """Update suspicion score based on detected behaviors"""
        # Behavior weights
        weights = {
            'POCKET_CONCEALMENT': 35,
            'WAIST_HIDING': 40,
            'SHIRT_CONCEALMENT': 35,
            'QUICK_GRAB': 25,
            'NERVOUS_LOOKING': 15
        }
        
        # Calculate score increase
        increase = sum(weights.get(b, 0) for b in behaviors)
        
        if increase > 0:
            person.suspicion_score = min(100, person.suspicion_score + increase * 0.5)
        else:
            # Decay when no suspicious behavior
            person.suspicion_score = max(0, person.suspicion_score - 0.3)
    
    def _update_state(self, person):
        """Update person's state based on suspicion score"""
        score = person.suspicion_score
        
        if score >= self.config['ALERT_THRESHOLD']:
            person.state = "ALERT"
        elif score >= self.config['SUSPICIOUS_THRESHOLD']:
            person.state = "SUSPICIOUS"
        elif score >= self.config['WATCHING_THRESHOLD']:
            person.state = "WATCHING"
        else:
            person.state = "NORMAL"
    
    def _generate_alert(self, person, bbox, frame, behaviors):
        """Generate shoplifting alert"""
        x1, y1, x2, y2 = map(int, bbox)
        
        behavior_str = ", ".join(behaviors) if behaviors else "Multiple indicators"
        
        print("\n" + "🚨" * 30)
        print(f"🚨 SHOPLIFTING ALERT! Person ID: {person.track_id}")
        print(f"🚨 Behaviors: {behavior_str}")
        print(f"🚨 Suspicion Score: {person.suspicion_score:.1f}%")
        print("🚨" * 30 + "\n")
        
        return {
            'timestamp': time.time(),
            'track_id': person.track_id,
            'behaviors': behaviors,
            'suspicion_score': person.suspicion_score,
            'bbox': bbox,
            'frame': frame.copy()
        }
    
    def _draw_person(self, frame, person, bbox, keypoints):
        """Draw person with skeleton and status"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color based on state
        colors = {
            "NORMAL": (0, 255, 0),      # Green
            "WATCHING": (0, 255, 255),   # Yellow
            "SUSPICIOUS": (0, 165, 255), # Orange
            "ALERT": (0, 0, 255)         # Red
        }
        color = colors.get(person.state, (0, 255, 0))
        
        # Draw bounding box
        thickness = 2 if person.state == "NORMAL" else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw skeleton
        self._draw_skeleton(frame, keypoints, color)
        
        # Draw status label
        label = f"ID:{person.track_id} [{person.state}]"
        score_label = f"Risk: {person.suspicion_score:.0f}%"
        
        # Label background
        cv2.rectangle(frame, (x1, y1-50), (x1 + 180, y1), color, -1)
        cv2.putText(frame, label, (x1+5, y1-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(frame, score_label, (x1+5, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # Draw suspicion bar
        bar_width = int((person.suspicion_score / 100) * 150)
        cv2.rectangle(frame, (x1, y2+5), (x1 + 150, y2+15), (50,50,50), -1)
        cv2.rectangle(frame, (x1, y2+5), (x1 + bar_width, y2+15), color, -1)
        
        # Show current behaviors
        if person.current_behaviors:
            behavior_text = " | ".join(person.current_behaviors[:2])
            cv2.putText(frame, behavior_text, (x1, y2+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def _draw_skeleton(self, frame, keypoints, color):
        """Draw pose skeleton"""
        skeleton_pairs = [
            (5, 6),   # Shoulders
            (5, 7), (7, 9),   # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12), # Torso
            (11, 12),         # Hips
            (11, 13), (13, 15), # Left leg
            (12, 14), (14, 16)  # Right leg
        ]
        
        conf = self.config['MIN_KEYPOINT_CONF']
        
        # Draw lines
        for start_idx, end_idx in skeleton_pairs:
            start_kp = keypoints[start_idx]
            end_kp = keypoints[end_idx]
            
            if start_kp[2] > conf and end_kp[2] > conf:
                start = (int(start_kp[0]), int(start_kp[1]))
                end = (int(end_kp[0]), int(end_kp[1]))
                cv2.line(frame, start, end, color, 2)
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > conf:
                # Highlight hands in yellow
                if i in [9, 10]:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 6, (0, 255, 255), -1)
                else:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)
    
    def _draw_ui(self, frame, active_persons):
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (350, 90), (0, 0, 0), -1)
        cv2.putText(frame, "LiftProof.Ai v2.0", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracking: {len(active_persons)} person(s)", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Alerts: {self.stats['total_alerts']} | Frames: {self.stats['frames_processed']}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection legend (bottom)
        cv2.rectangle(frame, (0, h-70), (280, h), (0, 0, 0), -1)
        cv2.putText(frame, "Status Colors:", (10, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "GREEN=Normal  YELLOW=Watching", (10, h-35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "ORANGE=Suspicious  RED=ALERT!", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit", (w-130, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def _get_body_height(self, keypoints):
        """Calculate approximate body height"""
        conf = self.config['MIN_KEYPOINT_CONF']
        
        shoulder = keypoints[KEYPOINTS['left_shoulder']]
        hip = keypoints[KEYPOINTS['left_hip']]
        
        if shoulder[2] > conf and hip[2] > conf:
            torso = self._distance(shoulder, hip)
            return torso * 2.5
        
        return 200  # Default
    
    def _distance(self, p1, p2):
        """Euclidean distance between two points"""
        if isinstance(p1, (list, np.ndarray)) and len(p1) >= 2:
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return 0
    
    def get_stats(self):
        """Return detection statistics"""
        return self.stats


def main():
    """Main function to run shoplifting detection"""
    print("\n" + "=" * 60)
    print("🛡️  LiftProof.Ai - Advanced Shoplifting Detection")
    print("=" * 60)
    
    # Initialize detector
    detector = AdvancedShopliftingDetector('yolov8n-pose.pt')
    
    # Open webcam
    print("\n📹 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access webcam!")
        print("Go to: System Preferences → Security & Privacy → Camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✅ Webcam ready!")
    print("\n" + "=" * 60)
    print("🎯 DETECTION ACTIVE")
    print("=" * 60)
    print("\n📋 Behaviors being monitored:")
    print("   • Hand in pocket (concealment)")
    print("   • Hand at waistband (hiding)")
    print("   • Hand under shirt (concealment)")
    print("   • Quick grabbing motion")
    print("   • Looking around nervously")
    print("\n👉 TEST: Put hand in pocket and HOLD for 1-2 seconds")
    print("👉 TEST: Move hand to waistband area")
    print("👉 TEST: Look around quickly left and right")
    print("\nPress 'Q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Process frame
        annotated_frame, alerts = detector.process_frame(frame)
        
        # Flash red on alert
        if alerts:
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            annotated_frame = cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0)
        
        # Display
        cv2.imshow('LiftProof.Ai - Shoplifting Detection', annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    stats = detector.get_stats()
    print("\n" + "=" * 60)
    print("📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"Frames Processed: {stats['frames_processed']}")
    print(f"Total Alerts: {stats['total_alerts']}")
    print("\nBehaviors Detected:")
    for behavior, count in stats['behaviors_detected'].items():
        if count > 0:
            print(f"   • {behavior}: {count}")
    print("=" * 60)
    print("✅ LiftProof.Ai stopped")


if __name__ == "__main__":
    main()