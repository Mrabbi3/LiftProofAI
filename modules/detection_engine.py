"""
Shoplifting Detection Engine
Analyzes pose data to detect suspicious behavior
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import math

class Person:
    """Tracked person with behavioral state"""
    
    def __init__(self, track_id):
        self.track_id = track_id
        self.suspicion_score = 0.0
        self.state = 0  # 0=Normal, 1=Suspicious, 2=Alert
        self.last_alert_time = 0
        self.first_seen = time.time()
    
    def get_person_height(self, keypoints):
        """Estimate person height"""
        nose = keypoints[0]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        ankle = left_ankle if left_ankle[2] > right_ankle[2] else right_ankle
        
        if nose[2] > 0.3 and ankle[2] > 0.3:
            height = math.sqrt((nose[0] - ankle[0])**2 + (nose[1] - ankle[1])**2)
            return max(height, 100)
        return 200


class ShopliftingDetector:
    """Detects shoplifting through behavioral analysis"""
    
    def __init__(self, model_path='yolov8n-pose.pt', config=None):
        self.config = config or {}
        
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        self.tracked_persons = {}
        
        # Detection thresholds
        self.pocket_distance_threshold = self.config.get('POCKET_DISTANCE_THRESHOLD', 0.15)
        self.pocket_time_threshold = self.config.get('POCKET_TIME_THRESHOLD', 45)
        self.confidence_threshold = self.config.get('CONFIDENCE_THRESHOLD', 0.5)
        
        self.total_frames_processed = 0
        self.total_alerts = 0
        
        # Keypoint indices
        self.KEYPOINT_LEFT_WRIST = 9
        self.KEYPOINT_RIGHT_WRIST = 10
        self.KEYPOINT_LEFT_HIP = 11
        self.KEYPOINT_RIGHT_HIP = 12
    
    def process_frame(self, frame):
        """Process frame and detect shoplifting behavior"""
        self.total_frames_processed += 1
        alerts = []
        
        # Run pose detection with tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return frame, alerts
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id
        
        if track_ids is None:
            return frame, alerts
        
        track_ids = track_ids.cpu().numpy().astype(int)
        keypoints = results[0].keypoints.data.cpu().numpy()
        
        # Process each person
        for i, track_id in enumerate(track_ids):
            bbox = boxes[i]
            kpts = keypoints[i]
            
            # Initialize if new
            if track_id not in self.tracked_persons:
                self.tracked_persons[track_id] = Person(track_id)
            
            person = self.tracked_persons[track_id]
            
            # Analyze behavior
            is_suspicious, suspicion_type = self._analyze_behavior(person, kpts)
            
            if is_suspicious:
                person.suspicion_score += 1
                
                # Check if threshold exceeded
                if person.suspicion_score >= self.pocket_time_threshold:
                    if time.time() - person.last_alert_time > 60:
                        alert = self._generate_alert(person, bbox, frame, suspicion_type)
                        alerts.append(alert)
                        person.last_alert_time = time.time()
                        person.state = 2
                        self.total_alerts += 1
                    else:
                        person.state = 1
                else:
                    person.state = 1
            else:
                person.suspicion_score = max(0, person.suspicion_score - 0.5)
                if person.suspicion_score == 0:
                    person.state = 0
            
            # Draw on frame
            frame = self._draw_person(frame, person, bbox, kpts)
        
        return frame, alerts
    
    def _analyze_behavior(self, person, keypoints):
        """Analyze for suspicious hand-to-pocket movements"""
        left_wrist = keypoints[self.KEYPOINT_LEFT_WRIST]
        right_wrist = keypoints[self.KEYPOINT_RIGHT_WRIST]
        left_hip = keypoints[self.KEYPOINT_LEFT_HIP]
        right_hip = keypoints[self.KEYPOINT_RIGHT_HIP]
        
        left_valid = left_wrist[2] > 0.3 and left_hip[2] > 0.3
        right_valid = right_wrist[2] > 0.3 and right_hip[2] > 0.3
        
        if not left_valid and not right_valid:
            return False, "none"
        
        person_height = person.get_person_height(keypoints)
        
        # Check left hand to left hip
        if left_valid:
            distance_left = self._euclidean_distance(left_wrist, left_hip)
            normalized_distance_left = distance_left / person_height
            
            if normalized_distance_left < self.pocket_distance_threshold:
                return True, "pocket_left"
        
        # Check right hand to right hip
        if right_valid:
            distance_right = self._euclidean_distance(right_wrist, right_hip)
            normalized_distance_right = distance_right / person_height
            
            if normalized_distance_right < self.pocket_distance_threshold:
                return True, "pocket_right"
        
        return False, "none"
    
    def _euclidean_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _generate_alert(self, person, bbox, frame, suspicion_type):
        """Generate alert"""
        x1, y1, x2, y2 = map(int, bbox)
        person_crop = frame[y1:y2, x1:x2]
        
        alert = {
            'timestamp': time.time(),
            'track_id': person.track_id,
            'suspicion_score': person.suspicion_score,
            'suspicion_type': suspicion_type,
            'person_image': person_crop,
            'full_frame': frame.copy()
        }
        
        print(f"\n🚨 SHOPLIFTING ALERT! Person {person.track_id} - {suspicion_type}")
        return alert
    
    def _draw_person(self, frame, person, bbox, keypoints):
        """Draw with color-coded state"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color based on state
        if person.state == 0:
            color = (0, 255, 0)  # Green - Normal
            label = "Normal"
        elif person.state == 1:
            color = (0, 255, 255)  # Yellow - Suspicious
            label = "SUSPICIOUS"
        else:
            color = (0, 0, 255)  # Red - ALERT!
            label = "🚨 SHOPLIFTING"
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label_text = f"ID:{person.track_id} {label} ({person.suspicion_score:.0f})"
        cv2.putText(frame, label_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw skeleton
        self._draw_skeleton(frame, keypoints, color)
        
        return frame
    
    def _draw_skeleton(self, frame, keypoints, color):
        """Draw pose skeleton"""
        # Skeleton connections
        skeleton = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for start_idx, end_idx in skeleton:
            start_pt = keypoints[start_idx]
            end_pt = keypoints[end_idx]
            
            if start_pt[2] > 0.3 and end_pt[2] > 0.3:
                start = (int(start_pt[0]), int(start_pt[1]))
                end = (int(end_pt[0]), int(end_pt[1]))
                cv2.line(frame, start, end, color, 2)
        
        # Draw keypoints
        for kpt in keypoints:
            if kpt[2] > 0.3:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 4, color, -1)
        
        return frame
    
    def get_stats(self):
        return {
            'frames_processed': self.total_frames_processed,
            'active_tracks': len(self.tracked_persons),
            'total_alerts': self.total_alerts
        }
