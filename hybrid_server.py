"""
Hybrid Multi-Camera Tracking Server
Combines per-camera Deep SORT/ByteTrack with global ReID and CLIP identification
"""

from flask import Flask, request, jsonify
import base64
import cv2
import torch
import numpy as np
import time
import sys
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
from PIL import Image

# Kalman Filter and Hungarian Algorithm imports
try:
    from filterpy.kalman import KalmanFilter
    from scipy.optimize import linear_sum_assignment
    ADVANCED_TRACKING_AVAILABLE = True
    print("✅ Kalman Filter and Hungarian Algorithm available")
except ImportError:
    ADVANCED_TRACKING_AVAILABLE = False
    print("⚠️ Advanced tracking not available. Install: pip install filterpy scipy")

# Import existing components
from Alignedreid_demo import Aligned_Reid_class
from clip_person_selector import CLIPPersonIdentifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize global components
ReId = Aligned_Reid_class()
clip_identifier = CLIPPersonIdentifier()

# Load YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.to(device)

@dataclass
class TrackResult:
    """Represents a single track result"""
    camera_id: str
    local_track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    is_new: bool
    reid_features: Optional[torch.Tensor] = None
    global_target_id: Optional[int] = None

class KalmanTracker:
    """
    Enhanced Kalman Filter tracker similar to ReTrack-VLM approach
    """
    count = 0
    
    def __init__(self, bbox, frame_idx=0):
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        
        # Initialize Kalman Filter
        if ADVANCED_TRACKING_AVAILABLE:
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            # State vector: [x, y, s, r, vx, vy, vs] where:
            # x, y: center coordinates
            # s: scale (area)
            # r: aspect ratio (width/height)
            # vx, vy, vs: velocities
            
            # Transition matrix (constant velocity model)
            self.kf.F = np.array([
                [1,0,0,0,1,0,0],
                [0,1,0,0,0,1,0],
                [0,0,1,0,0,0,1],
                [0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0],
                [0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1]
            ])
            
            # Measurement matrix
            self.kf.H = np.array([
                [1,0,0,0,0,0,0],
                [0,1,0,0,0,0,0],
                [0,0,1,0,0,0,0],
                [0,0,0,1,0,0,0]
            ])
            
            # Measurement noise
            self.kf.R[2:,2:] *= 10.
            
            # Initial state uncertainty
            self.kf.P[4:,4:] *= 1000.  # High uncertainty for velocities
            self.kf.P *= 10.
            
            # Process noise
            self.kf.Q[-1,-1] *= 0.01
            self.kf.Q[4:,4:] *= 0.01
            
            # Initialize state
            self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        else:
            # Fallback to simple position tracking
            self.position = bbox.copy()
            
        # Tracking state
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_frame_idx = frame_idx
        
        # Feature galleries for ReID
        self.reid_features = deque(maxlen=10)  # Increased gallery size
        self.clip_features = deque(maxlen=5)
        
        # ReID feature quality tracking
        self.reid_quality_scores = deque(maxlen=10)
        self.avg_reid_quality = 0.0
        
        # Association history for learning
        self.association_history = deque(maxlen=20)
        self.successful_associations = 0
        self.failed_associations = 0
        
        # Motion prediction
        self.velocity_history = deque(maxlen=10)
        self.trajectory = deque(maxlen=20)
        
        # Track quality metrics
        self.confidence_scores = deque(maxlen=10)
        self.avg_confidence = 1.0
        
        # State management
        self.state = 'ACTIVE'  # ACTIVE, LOST, RECOVERING
        self.lost_frames = 0
        self.max_lost_frames = 30
        
    def predict(self):
        """Predict next state using Kalman filter"""
        if ADVANCED_TRACKING_AVAILABLE:
            # Handle negative scale
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0
            
            self.kf.predict()
        
        self.age += 1
        self.time_since_update += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        # Update motion history
        current_bbox = self.get_state()
        if len(current_bbox) > 0:
            cx, cy, w, h = self._bbox_to_center_size(current_bbox[0])
            self.trajectory.append((cx, cy, w, h))
            
            # Calculate velocity
            if len(self.trajectory) >= 2:
                prev_pos = self.trajectory[-2]
                curr_pos = self.trajectory[-1]
                vx = curr_pos[0] - prev_pos[0]
                vy = curr_pos[1] - prev_pos[1]
                self.velocity_history.append((vx, vy))
        
        return self.get_state()
    
    def update(self, bbox, reid_feature=None, clip_feature=None, confidence=1.0, reid_quality=1.0):
        """Update tracker with new detection and enhanced feature management"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.lost_frames = 0
        
        if ADVANCED_TRACKING_AVAILABLE:
            self.kf.update(self.convert_bbox_to_z(bbox))
        else:
            self.position = bbox.copy()
            
        # Enhanced feature update with quality gating
        if reid_feature is not None and reid_quality > 0.5:  # Only update with good quality features
            self.reid_features.append(reid_feature)
            self.reid_quality_scores.append(reid_quality)
            
            # Update average quality
            if self.reid_quality_scores:
                self.avg_reid_quality = np.mean(self.reid_quality_scores)
                
        if clip_feature is not None:
            self.clip_features.append(clip_feature)
            
        # Update confidence
        self.confidence_scores.append(confidence)
        if self.confidence_scores:
            self.avg_confidence = np.mean(self.confidence_scores)
            
        # Update state
        if self.state == 'LOST':
            self.state = 'RECOVERING'
        elif self.state == 'RECOVERING' and self.hit_streak >= 3:
            self.state = 'ACTIVE'
            
        # Record successful association
        self.successful_associations += 1
        self.association_history.append(('success', confidence, reid_quality))
    
    def get_reid_confidence(self):
        """Get confidence in ReID features for this track"""
        if len(self.reid_features) == 0:
            return 0.0
        
        # Consider both feature count and quality
        feature_count_factor = min(1.0, len(self.reid_features) / 5.0)  # Normalize to [0,1]
        quality_factor = self.avg_reid_quality
        
        return feature_count_factor * quality_factor
    
    def should_use_reid_priority(self):
        """Determine if this track should prioritize ReID over IoU"""
        # Use ReID priority for tracks with good feature history
        return (self.get_reid_confidence() > 0.6 and 
                len(self.reid_features) >= 3 and
                self.avg_reid_quality > 0.7)
    
    def mark_missed(self):
        """Mark tracker as missed in current frame"""
        self.lost_frames += 1
        if self.lost_frames > self.max_lost_frames:
            self.state = 'LOST'
    
    def get_state(self):
        """Get current bounding box"""
        if ADVANCED_TRACKING_AVAILABLE:
            return self.convert_x_to_bbox(self.kf.x)
        else:
            return self.position.reshape((1, 4))
    
    def get_predicted_position(self, frames_ahead=1):
        """Get predicted position for future frames"""
        if not self.velocity_history:
            return self.get_state()[0]
            
        current_bbox = self.get_state()[0]
        cx, cy, w, h = self._bbox_to_center_size(current_bbox)
        
        # Average velocity
        if self.velocity_history:
            avg_vx = np.mean([v[0] for v in self.velocity_history])
            avg_vy = np.mean([v[1] for v in self.velocity_history])
            
            # Predict position
            pred_cx = cx + avg_vx * frames_ahead
            pred_cy = cy + avg_vy * frames_ahead
            
            # Convert back to bbox
            x1 = pred_cx - w/2
            y1 = pred_cy - h/2
            x2 = pred_cx + w/2
            y2 = pred_cy + h/2
            
            return np.array([x1, y1, x2, y2])
        
        return current_bbox
    
    def is_active(self):
        return self.state in ['ACTIVE', 'RECOVERING']
    
    def should_delete(self):
        return self.state == 'LOST' and self.lost_frames > self.max_lost_frames * 2
    
    @staticmethod
    def convert_bbox_to_z(bbox):
        """Convert bbox to measurement vector for Kalman filter"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h  # scale is area
        r = w / float(h) if h != 0 else 1  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def convert_x_to_bbox(x):
        """Convert state vector to bbox"""
        w = np.sqrt(abs(x[2] * x[3]))
        h = abs(x[2]) / w if w > 1e-6 else 0
        return np.array([
            x[0] - w/2., x[1] - h/2., 
            x[0] + w/2., x[1] + h/2.
        ]).reshape((1, 4))
    
    @staticmethod
    def _bbox_to_center_size(bbox):
        """Convert bbox to center and size"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w/2
        cy = bbox[1] + h/2
        return cx, cy, w, h


def compute_iou_matrix(detections, trackers):
    """Compute IoU matrix between detections and trackers"""
    if len(detections) == 0 or len(trackers) == 0:
        return np.empty((0, 0))
    
    iou_matrix = np.zeros((len(detections), len(trackers)))
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            trk_bbox = trk.get_state()[0]
            iou_matrix[d, t] = compute_iou(det[:4], trk_bbox)
    
    return iou_matrix

def compute_reid_similarity_matrix(reid_features, trackers):
    """Compute ReID similarity matrix between detections and trackers"""
    if len(reid_features) == 0 or len(trackers) == 0:
        return np.empty((0, 0))
    
    similarity_matrix = np.zeros((len(reid_features), len(trackers)))
    
    for d, det_feature in enumerate(reid_features):
        if det_feature is None:
            continue
            
        for t, tracker in enumerate(trackers):
            if len(tracker.reid_features) == 0:
                similarity_matrix[d, t] = 0.0
                continue
                
            # Compute average similarity with feature gallery
            similarities = []
            for track_feature in tracker.reid_features:
                if track_feature is not None and det_feature is not None:
                    try:
                        # Ensure tensors are on the same device
                        if det_feature.device != track_feature.device:
                            track_feature = track_feature.to(det_feature.device)
                        
                        similarity = torch.cosine_similarity(
                            det_feature.unsqueeze(0),
                            track_feature.unsqueeze(0)
                        ).item()
                        similarities.append(similarity)
                    except Exception as e:
                        logger.warning(f"ReID similarity computation failed: {e}")
                        continue
            
            if similarities:
                # Use maximum similarity from gallery for robustness
                similarity_matrix[d, t] = max(similarities)
            else:
                similarity_matrix[d, t] = 0.0
    
    return similarity_matrix

def compute_combined_cost_matrix(detections, trackers, reid_features, crowd_density='medium'):
    """
    Compute combined cost matrix using IoU and ReID features
    Adaptively weights based on crowd density
    """
    if len(detections) == 0 or len(trackers) == 0:
        empty_matrix = np.empty((0, 0))
        return empty_matrix, empty_matrix, empty_matrix
    
    # Compute individual matrices
    iou_matrix = compute_iou_matrix(detections, trackers)
    reid_matrix = compute_reid_similarity_matrix(reid_features, trackers)
    
    # Adaptive weighting based on crowd density
    if crowd_density == 'low':
        iou_weight = 0.7
        reid_weight = 0.3
    elif crowd_density == 'medium':
        iou_weight = 0.5
        reid_weight = 0.5
    else:  # high crowd density
        iou_weight = 0.3
        reid_weight = 0.7
    
    # Convert to cost matrices (1 - score for minimization)
    iou_cost = 1 - iou_matrix
    reid_cost = 1 - reid_matrix
    
    # Combined cost matrix
    combined_cost = iou_weight * iou_cost + reid_weight * reid_cost
    
    return combined_cost, iou_matrix, reid_matrix


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def associate_detections_to_trackers(detections, trackers, reid_features=None, 
                                    iou_threshold=0.3, reid_threshold=0.6, crowd_density='medium'):
    """
    Enhanced Hungarian algorithm-based association with ReID integration
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))
    
    # Compute combined cost matrix with adaptive weighting
    combined_cost, iou_matrix, reid_matrix = compute_combined_cost_matrix(
        detections, trackers, reid_features, crowd_density
    )
    
    # Set high cost for associations that fail both IoU and ReID thresholds
    for d in range(len(detections)):
        for t in range(len(trackers)):
            iou_score = iou_matrix[d, t] if len(iou_matrix) > 0 else 0
            reid_score = reid_matrix[d, t] if len(reid_matrix) > 0 else 0
            
            # Association is valid if either IoU OR ReID passes threshold
            # This allows ReID to rescue low IoU associations in crowds
            iou_valid = iou_score >= iou_threshold
            reid_valid = reid_score >= reid_threshold
            
            if not (iou_valid or reid_valid):
                combined_cost[d, t] = 1e5  # Set very high cost
    
    # Hungarian algorithm for optimal assignment
    if ADVANCED_TRACKING_AVAILABLE:
        matched_indices = linear_sum_assignment(combined_cost)
        matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
    else:
        # Fallback to greedy matching with combined scores
        matched_indices = []
        used_detections = set()
        used_trackers = set()
        
        # Create score matrix (higher is better)
        score_matrix = 1 - combined_cost
        
        # Sort by combined score descending
        all_matches = []
        for d in range(len(detections)):
            for t in range(len(trackers)):
                if combined_cost[d, t] < 1e5:  # Only consider valid associations
                    all_matches.append((d, t, score_matrix[d, t]))
        
        all_matches.sort(key=lambda x: x[2], reverse=True)
        
        for d, t, score in all_matches:
            if d not in used_detections and t not in used_trackers:
                matched_indices.append([d, t])
                used_detections.add(d)
                used_trackers.add(t)
        
        matched_indices = np.array(matched_indices) if matched_indices else np.empty((0, 2), dtype=int)
    
    # Filter matches by final validation
    if len(matched_indices) > 0:
        good_matches = []
        for match in matched_indices:
            d, t = match[0], match[1]
            if combined_cost[d, t] < 1e5:  # Valid association
                good_matches.append(match)
        matched_indices = np.array(good_matches) if good_matches else np.empty((0, 2), dtype=int)
    
    # Get unmatched detections and trackers
    matched_det_indices = matched_indices[:, 0] if len(matched_indices) > 0 else np.empty((0,), dtype=int)
    matched_trk_indices = matched_indices[:, 1] if len(matched_indices) > 0 else np.empty((0,), dtype=int)
    
    unmatched_detections = np.array([d for d in range(len(detections)) if d not in matched_det_indices])
    unmatched_trackers = np.array([t for t in range(len(trackers)) if t not in matched_trk_indices])
    
    return matched_indices, unmatched_detections, unmatched_trackers


class EnhancedDeepSortTracker:
    """
    Enhanced Deep SORT-style tracker with ReID-integrated association
    """
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.trackers = []
        self.frame_count = 0
        self.max_age = 30
        self.min_hits = 3
        self.iou_threshold = 0.3
        self.reid_threshold = 0.6
        
        # Crowd density estimation
        self.recent_detection_counts = deque(maxlen=30)  # Last 30 frames
        self.crowd_density = 'medium'
        
    def estimate_crowd_density(self, num_detections):
        """Estimate crowd density based on recent detection patterns"""
        self.recent_detection_counts.append(num_detections)
        
        if len(self.recent_detection_counts) < 10:
            return self.crowd_density
        
        avg_detections = np.mean(self.recent_detection_counts)
        
        # Dynamic thresholds based on camera setup
        if avg_detections <= 3:
            self.crowd_density = 'low'
        elif avg_detections <= 8:
            self.crowd_density = 'medium'
        else:
            self.crowd_density = 'high'
            
        return self.crowd_density
    
    def compute_reid_quality(self, reid_feature, bbox_area):
        """Estimate quality of extracted ReID feature"""
        if reid_feature is None:
            return 0.0
        
        # Feature norm as quality indicator
        feature_norm = torch.norm(reid_feature).item()
        
        # Bbox area as quality factor (larger area = better feature)
        area_factor = min(1.0, bbox_area / 10000.0)  # Normalize
        
        # Combine factors
        quality = min(1.0, feature_norm * area_factor)
        
        return quality
        
    def update(self, detections: np.ndarray, reid_features: List[torch.Tensor]) -> List[TrackResult]:
        """Enhanced update with ReID-integrated association"""
        self.frame_count += 1
        
        # Estimate crowd density
        crowd_density = self.estimate_crowd_density(len(detections))
        
        # Predict existing trackers
        for tracker in self.trackers:
            tracker.predict()
        
        # Enhanced association with ReID integration
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, self.trackers, reid_features, 
            self.iou_threshold, self.reid_threshold, crowd_density
        )
        
        # Update matched trackers with enhanced feature management
        for match in matched:
            det_idx, trk_idx = match[0], match[1]
            detection = detections[det_idx]
            reid_feature = reid_features[det_idx] if det_idx < len(reid_features) else None
            confidence = detection[4] if len(detection) > 4 else 1.0
            
            # Compute ReID quality
            bbox_area = (detection[2] - detection[0]) * (detection[3] - detection[1])
            reid_quality = self.compute_reid_quality(reid_feature, bbox_area)
            
            self.trackers[trk_idx].update(
                detection[:4], 
                reid_feature=reid_feature, 
                confidence=confidence,
                reid_quality=reid_quality
            )
        
        # Mark unmatched trackers as missed
        for trk_idx in unmatched_trks:
            self.trackers[trk_idx].mark_missed()
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            reid_feature = reid_features[det_idx] if det_idx < len(reid_features) else None
            
            new_tracker = KalmanTracker(detection[:4], self.frame_count)
            if reid_feature is not None:
                bbox_area = (detection[2] - detection[0]) * (detection[3] - detection[1])
                reid_quality = self.compute_reid_quality(reid_feature, bbox_area)
                new_tracker.reid_features.append(reid_feature)
                new_tracker.reid_quality_scores.append(reid_quality)
                new_tracker.avg_reid_quality = reid_quality
            
            self.trackers.append(new_tracker)
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if not t.should_delete()]
        
        # Convert to TrackResult format with ReID confidence
        results = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits or tracker.time_since_update <= 1:
                bbox = tracker.get_state()[0]
                
                result = TrackResult(
                    camera_id=self.camera_id,
                    local_track_id=tracker.id,
                    bbox=bbox.tolist(),
                    confidence=tracker.avg_confidence,
                    is_new=tracker.hits == 1,
                    reid_features=tracker.reid_features[-1] if tracker.reid_features else None
                )
                results.append(result)
        
        return results
        
class PerCameraTracker:
    """Per-camera tracker using Deep SORT or ByteTrack"""
    
    def __init__(self, camera_id: str, tracker_type: str = 'deepsort'):
        self.camera_id = camera_id
        self.tracker_type = tracker_type
        self.local_tracks = {}  # Maps local track IDs to global target IDs
        
        if tracker_type == 'deepsort':
            self.tracker = EnhancedDeepSortTracker(camera_id)
        else:
            # In production, add ByteTrack implementation
            self.tracker = EnhancedDeepSortTracker(camera_id)  # Fallback for now
            
        logger.info(f"Initialized {tracker_type} tracker for camera {camera_id}")
    
    def process_frame(self, frame: np.ndarray, detections: np.ndarray) -> List[TrackResult]:
        """Process frame and return tracking results"""
        try:
            # Extract ReID features for detections
            reid_features = []
            frame_height, frame_width = frame.shape[:2]
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(x1 + 1, min(x2, frame_width))
                y2 = max(y1 + 1, min(y2, frame_height))
                
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size > 0:
                    try:
                        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(person_crop_rgb)
                        features = ReId.get_features(pil_image)
                        reid_features.append(features)
                    except Exception as crop_error:
                        logger.warning(f"Failed to extract ReID features: {crop_error}")
                        reid_features.append(None)
                else:
                    reid_features.append(None)
            
            # Update tracker
            tracks = self.tracker.update(detections, reid_features)
            
            # Add global target associations
            for track in tracks:
                if track.local_track_id in self.local_tracks:
                    track.global_target_id = self.local_tracks[track.local_track_id]
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error processing frame for camera {self.camera_id}: {e}")
            return []

class GlobalMultiCameraManager:
    """Global manager for multi-camera tracking with CLIP and ReID"""
    
    def __init__(self):
        self.camera_trackers: Dict[str, PerCameraTracker] = {}
        self.global_targets: Dict[int, Dict] = {}
        self.next_target_id = 1
        self.reid_threshold = 0.6
        self.clip_threshold = 0.25
        
        logger.info("Initialized Global Multi-Camera Manager")
    
    def add_camera(self, camera_id: str, tracker_type: str = 'deepsort'):
        """Add a new camera with specified tracker type"""
        self.camera_trackers[camera_id] = PerCameraTracker(camera_id, tracker_type)
        logger.info(f"Added camera {camera_id} with {tracker_type} tracker")
    
    def add_clip_target(self, description: str, target_name: str = None) -> int:
        """Add a new CLIP target description"""
        target_id = self.next_target_id
        self.next_target_id += 1
        
        if not target_name:
            target_name = f"Target_{target_id}"
        
        self.global_targets[target_id] = {
            'name': target_name,
            'clip_description': description,
            'reid_features': None,
            'active_cameras': {},  # camera_id: local_track_id
            'last_seen': None,
            'first_detected': time.time(),
            'confidence': 0.0
        }
        
        # Add to CLIP identifier
        clip_identifier.add_target_description(target_id, description)
        
        logger.info(f"Added CLIP target {target_id}: '{description}'")
        return target_id
    
    def process_all_cameras(self, camera_frames: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process all cameras and return comprehensive results"""
        all_camera_results = {}
        
        # Step 1: Process each camera independently
        for camera_id, frame in camera_frames.items():
            if camera_id not in self.camera_trackers:
                logger.warning(f"No tracker found for camera {camera_id}")
                continue
                
            try:
                # Run YOLO detection
                results = model(frame)
                
                # Check if results are valid
                if results is None or len(results.pandas().xyxy) == 0:
                    all_camera_results[camera_id] = []
                    continue
                    
                detections = results.pandas().xyxy[0].values
                
                # Validate detections format
                if len(detections) == 0 or detections.shape[1] < 6:
                    all_camera_results[camera_id] = []
                    continue
                
                # Filter for person detections
                person_detections = detections[detections[:, 5] == 0]  # class 0 = person
                
                if len(person_detections) > 0:
                    # Process with camera tracker
                    camera_results = self.camera_trackers[camera_id].process_frame(
                        frame, person_detections
                    )
                    all_camera_results[camera_id] = camera_results
                else:
                    all_camera_results[camera_id] = []
                    
            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {e}")
                all_camera_results[camera_id] = []
        
        # Step 2: Global association across cameras
        self.associate_across_cameras(all_camera_results, camera_frames)
        
        # Step 3: Build comprehensive response
        return self.build_response(all_camera_results)
    
    def associate_across_cameras(self, all_camera_results: Dict[str, List[TrackResult]], 
                                camera_frames: Dict[str, np.ndarray]):
        """Handle global associations across cameras"""
        
        # Handle new tracks with CLIP identification
        for camera_id, camera_results in all_camera_results.items():
            frame = camera_frames[camera_id]
            
            for track_result in camera_results:
                if track_result.is_new and track_result.global_target_id is None:
                    self.handle_new_track(track_result, frame)
        
        # Handle cross-camera ReID associations
        self.handle_cross_camera_reid(all_camera_results, camera_frames)
        
        # Clean up old targets
        self.cleanup_old_targets()
    
    def handle_new_track(self, track_result: TrackResult, frame: np.ndarray):
        """Handle new track detection with CLIP identification"""
        try:
            # Try CLIP identification
            clip_match = clip_identifier.identify_new_person(
                frame, track_result.bbox, confidence_threshold=self.clip_threshold
            )
            
            if clip_match:
                target_id, similarity_score, description = clip_match
                
                # Associate local track with global target
                if target_id in self.global_targets:
                    # Update existing target
                    target = self.global_targets[target_id]
                    target['active_cameras'][track_result.camera_id] = track_result.local_track_id
                    target['last_seen'] = time.time()
                    target['confidence'] = max(target['confidence'], similarity_score)
                    
                    # Store ReID features if not already stored
                    if target['reid_features'] is None and track_result.reid_features is not None:
                        target['reid_features'] = track_result.reid_features
                    
                    # Update local association
                    self.camera_trackers[track_result.camera_id].local_tracks[track_result.local_track_id] = target_id
                    track_result.global_target_id = target_id
                    
                    logger.info(f"CLIP identified track {track_result.local_track_id} in camera {track_result.camera_id} as target {target_id} (score: {similarity_score:.3f})")
                    
        except Exception as e:
            logger.error(f"Error handling new track: {e}")
    
    def handle_cross_camera_reid(self, all_camera_results: Dict[str, List[TrackResult]], 
                                camera_frames: Dict[str, np.ndarray]):
        """Handle cross-camera associations using ReID"""
        
        # Collect unassociated tracks
        unassociated_tracks = []
        
        for camera_id, camera_results in all_camera_results.items():
            for track_result in camera_results:
                if (track_result.global_target_id is None and 
                    track_result.reid_features is not None):
                    unassociated_tracks.append(track_result)
        
        # Try to associate with existing global targets
        for track_result in unassociated_tracks:
            best_target_id = None
            best_reid_score = 0.0
            
            for target_id, target_data in self.global_targets.items():
                if target_data['reid_features'] is not None:
                    try:
                        # Ensure tensors are on the same device
                        track_features = track_result.reid_features
                        target_features = target_data['reid_features']
                        
                        if track_features.device != target_features.device:
                            target_features = target_features.to(track_features.device)
                        
                        reid_score = torch.cosine_similarity(
                            track_features.unsqueeze(0),
                            target_features.unsqueeze(0)
                        ).item()
                        
                        if reid_score > best_reid_score and reid_score > self.reid_threshold:
                            best_reid_score = reid_score
                            best_target_id = target_id
                    except Exception as e:
                        logger.warning(f"Cross-camera ReID comparison failed: {e}")
                        continue
            
            # Associate if good match found
            if best_target_id:
                target = self.global_targets[best_target_id]
                target['active_cameras'][track_result.camera_id] = track_result.local_track_id
                target['last_seen'] = time.time()
                
                # Update local association
                self.camera_trackers[track_result.camera_id].local_tracks[track_result.local_track_id] = best_target_id
                track_result.global_target_id = best_target_id
                
                logger.info(f"Cross-camera ReID: Associated camera {track_result.camera_id} track {track_result.local_track_id} with target {best_target_id} (score: {best_reid_score:.3f})")
    
    def cleanup_old_targets(self):
        """Remove old inactive targets"""
        current_time = time.time()
        timeout = 30.0  # 30 seconds timeout
        
        targets_to_remove = []
        for target_id, target_data in self.global_targets.items():
            if (target_data['last_seen'] and 
                current_time - target_data['last_seen'] > timeout):
                targets_to_remove.append(target_id)
        
        for target_id in targets_to_remove:
            # Remove from CLIP identifier
            clip_identifier.remove_target_description(target_id)
            
            # Remove from global targets
            del self.global_targets[target_id]
            
            # Remove from camera trackers
            for camera_tracker in self.camera_trackers.values():
                tracks_to_remove = [k for k, v in camera_tracker.local_tracks.items() if v == target_id]
                for track_id in tracks_to_remove:
                    del camera_tracker.local_tracks[track_id]
            
            logger.info(f"Removed inactive target {target_id}")
    
    def build_response(self, all_camera_results: Dict[str, List[TrackResult]]) -> Dict[str, Any]:
        """Build comprehensive response with enhanced tracking information"""
        response = {
            'hybrid_multi_camera_tracking': True,
            'timestamp': time.time(),
            'global_targets': [],
            'per_camera_results': {},
            'tracking_analytics': {
                'total_targets': len(self.global_targets),
                'active_cameras': len(all_camera_results),
                'total_tracks': sum(len(results) for results in all_camera_results.values()),
                'crowd_density_analysis': {}
            }
        }
        
        # Add global target information
        for target_id, target_data in self.global_targets.items():
            active_cameras = list(target_data['active_cameras'].keys())
            response['global_targets'].append({
                'target_id': target_id,
                'name': target_data['name'],
                'description': target_data['clip_description'],
                'active_cameras': active_cameras,
                'confidence': target_data['confidence'],
                'last_seen': target_data['last_seen'],
                'tracking_method': 'hybrid_reid_motion_clip'
            })
        
        # Add per-camera results with enhanced analytics
        for camera_id, camera_results in all_camera_results.items():
            # Get crowd density from camera tracker
            crowd_density = 'medium'  # Default
            if camera_id in self.camera_trackers:
                crowd_density = self.camera_trackers[camera_id].tracker.crowd_density
            
            tracks = []
            reid_enhanced_tracks = 0
            
            for track in camera_results:
                track_info = {
                    'local_track_id': track.local_track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'is_new': track.is_new,
                    'global_target_id': track.global_target_id,
                    'tracking_method': 'reid_enhanced_motion' if not track.is_new else 'new_detection'
                }
                
                if track.global_target_id:
                    target_data = self.global_targets[track.global_target_id]
                    track_info['target_name'] = target_data['name']
                    track_info['target_description'] = target_data['clip_description']
                    reid_enhanced_tracks += 1
                
                tracks.append(track_info)
            
            response['per_camera_results'][camera_id] = {
                'tracks': tracks,
                'total_tracks': len(tracks),
                'associated_targets': len([t for t in tracks if t['global_target_id']]),
                'crowd_density': crowd_density,
                'reid_enhanced_tracks': reid_enhanced_tracks
            }
            
            # Add to crowd density analysis
            response['tracking_analytics']['crowd_density_analysis'][camera_id] = {
                'density': crowd_density,
                'track_count': len(tracks),
                'reid_usage': f"{reid_enhanced_tracks}/{len(tracks)}" if tracks else "0/0"
            }
        
        return response

# Global hybrid tracker instance
hybrid_tracker = GlobalMultiCameraManager()

def decode_base64_image(encoded_string: str) -> Optional[np.ndarray]:
    """Decode base64 encoded image with validation"""
    try:
        if not encoded_string or not isinstance(encoded_string, str):
            return None
            
        img_data = base64.b64decode(encoded_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Validate decoded image
        if img is None or img.size == 0:
            logger.warning("Decoded image is empty or invalid")
            return None
            
        return img
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None

@app.route('/add_camera', methods=['POST'])
def add_camera():
    """Add a new camera to the tracking system"""
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        tracker_type = data.get('tracker_type', 'deepsort')
        
        if not camera_id:
            return jsonify({'error': 'camera_id is required'}), 400
        
        hybrid_tracker.add_camera(camera_id, tracker_type)
        
        return jsonify({
            'status': 'success',
            'message': f'Added camera {camera_id} with {tracker_type} tracker',
            'camera_id': camera_id,
            'tracker_type': tracker_type
        })
        
    except Exception as e:
        logger.error(f"Error adding camera: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_target', methods=['POST'])
def add_target():
    """Add a new CLIP target description"""
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        target_name = data.get('target_name', '').strip()
        
        if not description:
            return jsonify({'error': 'description is required'}), 400
        
        target_id = hybrid_tracker.add_clip_target(description, target_name)
        
        return jsonify({
            'status': 'success',
            'target_id': target_id,
            'target_name': hybrid_tracker.global_targets[target_id]['name'],
            'description': description,
            'message': f'Added target: {description}'
        })
        
    except Exception as e:
        logger.error(f"Error adding target: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_hybrid', methods=['POST'])
def predict_hybrid():
    """Main prediction endpoint for hybrid tracking"""
    try:
        data = request.get_json()
        
        if 'camera_frames' in data:
            # Multi-camera request
            camera_frames = {}
            
            for camera_id, encoded_frame in data['camera_frames'].items():
                frame = decode_base64_image(encoded_frame)
                if frame is not None:
                    camera_frames[camera_id] = frame
                else:
                    logger.warning(f"Failed to decode frame for camera {camera_id}")
            
            if not camera_frames:
                return jsonify({'error': 'No valid camera frames provided'}), 400
            
            # Process with hybrid tracker
            results = hybrid_tracker.process_all_cameras(camera_frames)
            return jsonify(results)
            
        elif 'camera_id' in data and 'image' in data:
            # Single camera request
            camera_id = data['camera_id']
            encoded_frame = data['image']
            
            frame = decode_base64_image(encoded_frame)
            if frame is None:
                return jsonify({'error': 'Failed to decode image'}), 400
            
            # Process single camera
            results = hybrid_tracker.process_all_cameras({camera_id: frame})
            return jsonify(results)
            
        else:
            return jsonify({'error': 'Invalid request format. Use camera_frames for multi-camera or camera_id+image for single camera'}), 400
            
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_status', methods=['GET'])
def get_status():
    """Get current system status with enhanced analytics"""
    try:
        status = {
            'system': 'Enhanced Hybrid Multi-Camera Tracking Server',
            'version': '2.0 - ReID Enhanced',
            'active_cameras': list(hybrid_tracker.camera_trackers.keys()),
            'total_targets': len(hybrid_tracker.global_targets),
            'tracking_features': {
                'reid_integrated_association': True,
                'adaptive_crowd_density': True,
                'feature_quality_gating': True,
                'multi_modal_fusion': True
            },
            'targets': [],
            'camera_analytics': {}
        }
        
        for target_id, target_data in hybrid_tracker.global_targets.items():
            status['targets'].append({
                'target_id': target_id,
                'name': target_data['name'],
                'description': target_data['clip_description'],
                'active_cameras': list(target_data['active_cameras'].keys()),
                'last_seen': target_data['last_seen']
            })
        
        # Add camera-specific analytics
        for camera_id, camera_tracker in hybrid_tracker.camera_trackers.items():
            tracker = camera_tracker.tracker
            status['camera_analytics'][camera_id] = {
                'crowd_density': tracker.crowd_density,
                'active_tracks': len(tracker.trackers),
                'tracking_mode': f"IoU+ReID ({tracker.crowd_density} density)",
                'reid_threshold': tracker.reid_threshold,
                'iou_threshold': tracker.iou_threshold
            }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_all', methods=['POST'])
def clear_all():
    """Clear all targets and reset system"""
    try:
        # Clear CLIP targets
        clip_identifier.clear_all_targets()
        
        # Clear global targets
        hybrid_tracker.global_targets.clear()
        hybrid_tracker.next_target_id = 1
        
        # Clear camera tracker associations
        for camera_tracker in hybrid_tracker.camera_trackers.values():
            camera_tracker.local_tracks.clear()
        
        return jsonify({
            'status': 'success',
            'message': 'All targets cleared'
        })
        
    except Exception as e:
        logger.error(f"Error clearing all: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Hybrid Multi-Camera Tracking Server v2.0...")
    print("📚 Enhanced Features:")
    print("  • ReID-Integrated Intra-Camera Tracking")
    print("  • Adaptive IoU + ReID Association")
    print("  • Dynamic Crowd Density Estimation")
    print("  • Feature Quality Gating & Galleries")
    print("  • Multi-Modal Fusion (Motion + Appearance)")
    print("  • CLIP-Focused New Target Detection")
    print("  • Enhanced Cross-Camera Association")
    print("  • Robust Occlusion Handling")
    print("\n🔗 API Endpoints:")
    print("  POST /add_camera       - Add camera with tracker type")
    print("  POST /add_target       - Add CLIP target description") 
    print("  POST /predict_hybrid   - Enhanced tracking endpoint")
    print("  GET  /get_status       - System status with analytics")
    print("  POST /clear_all        - Clear all targets")
    print("\n⚡ Improvements:")
    print("  • Better crowded scene performance")
    print("  • Reduced identity switches")
    print("  • Enhanced occlusion recovery")
    print("  • Adaptive algorithm weighting")
    print("\n🎯 Starting server on http://0.0.0.0:5001")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
