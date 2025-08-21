#!/usr/bin/env python3
"""
Standalone CLIP ReID Video Processor
Process video clips with text descriptions without Flask server
Handles multiple entities with consistent ID tracking
"""

import cv2
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandaloneCLIPReID:
    """
    Standalone CLIP ReID processor for video analysis
    """
    
    def __init__(self, device: str = None, clip_model: str = "ViT-B/32"):
        """
        Initialize the standalone processor
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            clip_model: CLIP model to use
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.clip_model.eval()
        
        # Load YOLO model
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.yolo_model.to(self.device)
        
        # Tracking variables
        self.person_tracks = {}  # track_id -> track_info
        self.next_track_id = 1
        self.feature_memory = defaultdict(list)  # track_id -> list of features
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # track_id -> bbox history
        
        # Configuration
        self.clip_threshold = 0.25
        self.reid_threshold = 0.7
        self.max_disappeared = 10
        self.feature_memory_size = 5
        
        logger.info(f"Initialized StandaloneCLIPReID on {self.device}")
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame using YOLO
        
        Args:
            frame: Input frame
            
        Returns:
            List of person detections
        """
        results = self.yolo_model(frame)
        detections = results.pandas().xyxy[0].values.tolist()
        
        # Filter for person class (class 0)
        person_detections = []
        for det in detections:
            if len(det) >= 6 and det[5] == 0 and det[4] > 0.5:  # person class with confidence > 0.5
                person_detections.append({
                    'bbox': [int(x) for x in det[:4]],  # x1, y1, x2, y2
                    'confidence': det[4],
                    'class': int(det[5])
                })
        
        return person_detections
    
    def extract_person_crops(self, frame: np.ndarray, detections: List[Dict]) -> List[Image.Image]:
        """
        Extract person crops from detections
        
        Args:
            frame: Input frame
            detections: Person detections
            
        Returns:
            List of PIL Images
        """
        crops = []
        h, w = frame.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                crops.append(pil_crop)
            else:
                crops.append(None)
        
        return crops
    
    def encode_text_query(self, text_description: str) -> torch.Tensor:
        """
        Encode text description using CLIP
        
        Args:
            text_description: Text description of target person
            
        Returns:
            Encoded text features
        """
        text_tokens = clip.tokenize([text_description]).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def encode_person_crops(self, crops: List[Image.Image]) -> torch.Tensor:
        """
        Encode person crops using CLIP
        
        Args:
            crops: List of person crop images
            
        Returns:
            Encoded image features
        """
        valid_crops = [crop for crop in crops if crop is not None]
        
        if not valid_crops:
            return torch.empty(0, 512).to(self.device)
        
        # Preprocess images
        image_tensors = torch.stack([self.clip_preprocess(crop) for crop in valid_crops]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def find_clip_matches(self, text_features: torch.Tensor, image_features: torch.Tensor, 
                         detections: List[Dict]) -> List[Tuple[int, float, Dict]]:
        """
        Find CLIP matches for text query
        
        Args:
            text_features: Encoded text features
            image_features: Encoded image features
            detections: Person detections
            
        Returns:
            List of (detection_index, similarity_score, detection)
        """
        if len(image_features) == 0:
            return []
        
        # Compute similarities
        similarities = torch.matmul(text_features, image_features.T).squeeze()
        
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)
        
        # Find matches above threshold
        matches = []
        for i, score in enumerate(similarities):
            score_val = score.item()
            if score_val >= self.clip_threshold:
                matches.append((i, score_val, detections[i]))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def compute_reid_features(self, crops: List[Image.Image]) -> List[np.ndarray]:
        """
        Compute ReID features for person crops
        Note: This is a simplified version. In practice, you'd use your trained ReID model
        
        Args:
            crops: Person crop images
            
        Returns:
            List of ReID feature vectors
        """
        reid_features = []
        
        for crop in crops:
            if crop is None:
                reid_features.append(None)
                continue
            
            # Use CLIP features as ReID features (simplified)
            # In practice, replace this with your actual ReID model
            crop_tensor = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.clip_model.encode_image(crop_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                reid_features.append(features.cpu().numpy().flatten())
        
        return reid_features
    
    def match_with_existing_tracks(self, reid_features: List[np.ndarray], 
                                 detections: List[Dict]) -> List[Optional[int]]:
        """
        Match detections with existing tracks using ReID features
        
        Args:
            reid_features: ReID features for current detections
            detections: Current detections
            
        Returns:
            List of track IDs (None for new tracks)
        """
        assignments = [None] * len(detections)
        
        for i, (reid_feat, detection) in enumerate(zip(reid_features, detections)):
            if reid_feat is None:
                continue
            
            best_match_id = None
            best_similarity = 0
            
            # Compare with existing tracks
            for track_id, track_features in self.feature_memory.items():
                if not track_features:
                    continue
                
                # Compute average similarity with track features
                similarities = []
                for track_feat in track_features:
                    sim = 1 - cosine(reid_feat, track_feat)
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > best_similarity and avg_similarity > self.reid_threshold:
                    best_similarity = avg_similarity
                    best_match_id = track_id
            
            assignments[i] = best_match_id
        
        return assignments
    
    def update_tracks(self, detections: List[Dict], reid_features: List[np.ndarray], 
                     assignments: List[Optional[int]], clip_matches: List[Tuple]) -> Dict[int, Dict]:
        """
        Update tracking information
        
        Args:
            detections: Current detections
            reid_features: ReID features
            assignments: Track assignments
            clip_matches: CLIP matches
            
        Returns:
            Updated track information
        """
        current_tracks = {}
        clip_match_indices = {match[0] for match in clip_matches}
        
        for i, (detection, reid_feat, track_id) in enumerate(zip(detections, reid_features, assignments)):
            if reid_feat is None:
                continue
            
            # Assign track ID
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
            
            # Update feature memory
            self.feature_memory[track_id].append(reid_feat)
            if len(self.feature_memory[track_id]) > self.feature_memory_size:
                self.feature_memory[track_id].pop(0)
            
            # Update track history
            bbox = detection['bbox']
            self.track_history[track_id].append(bbox)
            
            # Determine if this is a CLIP match
            is_clip_match = i in clip_match_indices
            clip_score = 0
            if is_clip_match:
                for match in clip_matches:
                    if match[0] == i:
                        clip_score = match[1]
                        break
            
            # Store track info
            current_tracks[track_id] = {
                'bbox': bbox,
                'confidence': detection['confidence'],
                'is_clip_match': is_clip_match,
                'clip_score': clip_score,
                'reid_feature': reid_feat,
                'disappeared': 0
            }
        
        # Update disappeared counter for missing tracks
        for track_id in list(self.person_tracks.keys()):
            if track_id not in current_tracks:
                if track_id in self.person_tracks:
                    self.person_tracks[track_id]['disappeared'] += 1
                    if self.person_tracks[track_id]['disappeared'] > self.max_disappeared:
                        # Remove track
                        del self.person_tracks[track_id]
                        if track_id in self.feature_memory:
                            del self.feature_memory[track_id]
                        if track_id in self.track_history:
                            del self.track_history[track_id]
        
        # Update person tracks
        self.person_tracks.update(current_tracks)
        
        return current_tracks
    
    def draw_tracks(self, frame: np.ndarray, tracks: Dict[int, Dict], 
                   text_query: str = "") -> np.ndarray:
        """
        Draw tracking information on frame
        
        Args:
            frame: Input frame
            tracks: Track information
            text_query: Text query used for CLIP matching
            
        Returns:
            Frame with tracking visualization
        """
        vis_frame = frame.copy()
        
        # Add text query at top
        if text_query:
            cv2.putText(vis_frame, f"Query: {text_query}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Color scheme
        colors = [
            (0, 255, 0),    # Green for CLIP matches
            (255, 0, 0),    # Blue for regular tracks
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]
        
        for track_id, track_info in tracks.items():
            bbox = track_info['bbox']
            x1, y1, x2, y2 = bbox
            
            # Choose color
            if track_info['is_clip_match']:
                color = (0, 255, 0)  # Green for CLIP matches
                thickness = 3
            else:
                color = colors[track_id % len(colors)]
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label_parts = [f"ID:{track_id}"]
            
            if track_info['is_clip_match']:
                label_parts.append(f"CLIP:{track_info['clip_score']:.2f}")
            
            label_parts.append(f"Conf:{track_info['confidence']:.2f}")
            label = " ".join(label_parts)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw track history (trajectory)
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = []
                for hist_bbox in self.track_history[track_id]:
                    center_x = (hist_bbox[0] + hist_bbox[2]) // 2
                    center_y = (hist_bbox[1] + hist_bbox[3]) // 2
                    points.append((center_x, center_y))
                
                # Draw trajectory
                for i in range(1, len(points)):
                    cv2.line(vis_frame, points[i-1], points[i], color, 2)
        
        # Add statistics
        clip_matches = sum(1 for track in tracks.values() if track['is_clip_match'])
        total_tracks = len(tracks)
        
        stats_text = f"Total Tracks: {total_tracks} | CLIP Matches: {clip_matches}"
        cv2.putText(vis_frame, stats_text, (10, vis_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def process_video(self, video_path: str, text_query: str, 
                     output_path: str = None, display: bool = True) -> Dict:
        """
        Process entire video with CLIP ReID tracking
        
        Args:
            video_path: Path to input video
            text_query: Text description of target person(s)
            output_path: Path to save output video
            display: Whether to display frames during processing
            
        Returns:
            Processing statistics
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        logger.info(f"Text query: '{text_query}'")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Encode text query
        text_features = self.encode_text_query(text_query)
        
        # Processing statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'clip_matches': 0,
            'unique_tracks': set(),
            'processing_time': 0
        }
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Detect persons
                detections = self.detect_persons(frame)
                stats['total_detections'] += len(detections)
                
                if detections:
                    # Extract crops
                    crops = self.extract_person_crops(frame, detections)
                    
                    # Encode crops with CLIP
                    image_features = self.encode_person_crops(crops)
                    
                    # Find CLIP matches
                    clip_matches = self.find_clip_matches(text_features, image_features, detections)
                    stats['clip_matches'] += len(clip_matches)
                    
                    # Compute ReID features
                    reid_features = self.compute_reid_features(crops)
                    
                    # Match with existing tracks
                    assignments = self.match_with_existing_tracks(reid_features, detections)
                    
                    # Update tracks
                    current_tracks = self.update_tracks(detections, reid_features, assignments, clip_matches)
                    stats['unique_tracks'].update(current_tracks.keys())
                else:
                    current_tracks = {}
                    clip_matches = []
                
                # Draw tracking visualization
                vis_frame = self.draw_tracks(frame, current_tracks, text_query)
                
                # Save frame
                if output_path:
                    out.write(vis_frame)
                
                # Display frame
                if display:
                    cv2.imshow('CLIP ReID Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                stats['processed_frames'] = frame_count
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / frame_count) * (total_frames - frame_count)
                    logger.info(f"Progress: {progress:.1f}% | ETA: {eta:.1f}s")
        
        finally:
            # Cleanup
            cap.release()
            if output_path:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        stats['processing_time'] = time.time() - start_time
        stats['unique_tracks'] = len(stats['unique_tracks'])
        stats['fps'] = stats['processed_frames'] / stats['processing_time']
        
        logger.info("Processing completed!")
        logger.info(f"Processed {stats['processed_frames']} frames in {stats['processing_time']:.2f}s")
        logger.info(f"Average FPS: {stats['fps']:.2f}")
        logger.info(f"Total detections: {stats['total_detections']}")
        logger.info(f"CLIP matches: {stats['clip_matches']}")
        logger.info(f"Unique tracks: {stats['unique_tracks']}")
        
        return stats

def create_demo_notebook():
    """
    Create a demo notebook content for easy usage
    """
    notebook_content = '''
# CLIP ReID Video Processing Demo

This notebook demonstrates how to use the standalone CLIP ReID processor for video analysis.

## Setup

```python
import cv2
import numpy as np
from standalone_clip_reid_notebook import StandaloneCLIPReID
import matplotlib.pyplot as plt
from IPython.display import Video, display

# Initialize the processor
processor = StandaloneCLIPReID(device="cuda")  # or "cpu"
```

## Basic Usage

```python
# Process a video with text description
video_path = "path/to/your/video.mp4"
text_query = "person wearing red shirt and blue jeans"
output_path = "output_tracked_video.mp4"

# Process the video
stats = processor.process_video(
    video_path=video_path,
    text_query=text_query,
    output_path=output_path,
    display=False  # Set to True to see real-time processing
)

# Display results
print("Processing Statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")
```

## Advanced Usage

```python
# Process with custom parameters
processor.clip_threshold = 0.3  # Adjust CLIP matching threshold
processor.reid_threshold = 0.8  # Adjust ReID matching threshold
processor.max_disappeared = 15  # Frames before track is removed

# Process multiple queries
queries = [
    "person wearing red shirt",
    "woman in blue dress", 
    "man with black jacket"
]

for i, query in enumerate(queries):
    output_file = f"output_query_{i+1}.mp4"
    stats = processor.process_video(video_path, query, output_file)
    print(f"Query '{query}': {stats['clip_matches']} matches found")
```

## Visualization

```python
# Display the output video
Video("output_tracked_video.mp4", width=800)
```

## Frame-by-Frame Processing

```python
# For more control, process frame by frame
cap = cv2.VideoCapture(video_path)
text_features = processor.encode_text_query("person wearing red shirt")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and process
    detections = processor.detect_persons(frame)
    if detections:
        crops = processor.extract_person_crops(frame, detections)
        image_features = processor.encode_person_crops(crops)
        clip_matches = processor.find_clip_matches(text_features, image_features, detections)
        
        # Your custom processing here
        print(f"Frame {frame_count}: {len(clip_matches)} CLIP matches")
    
    frame_count += 1

cap.release()
```

## Tips for Better Results

1. **Text Descriptions**: Be specific but not overly detailed
   - Good: "person wearing red shirt"
   - Better: "person wearing red shirt and blue jeans"
   - Avoid: "person wearing bright red cotton shirt with small logo"

2. **Threshold Tuning**:
   - Lower `clip_threshold` (0.15-0.25) for more matches
   - Higher `reid_threshold` (0.7-0.9) for better tracking consistency

3. **Video Quality**:
   - Higher resolution videos work better
   - Good lighting conditions improve results
   - Avoid heavily occluded scenes

4. **Multiple Entities**:
   - The system automatically handles multiple people with similar features
   - Each person gets a unique track ID
   - CLIP matches are highlighted in green
'''
    
    return notebook_content

# Example usage function
def demo_usage():
    """
    Demonstration of how to use the standalone processor
    """
    print("CLIP ReID Standalone Processor Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = StandaloneCLIPReID()
    
    # Example video processing (replace with actual video path)
    video_path = "sample_video.mp4"  # Replace with your video
    text_query = "person wearing red shirt"
    output_path = "tracked_output.mp4"
    
    if Path(video_path).exists():
        print(f"Processing video: {video_path}")
        print(f"Text query: '{text_query}'")
        
        stats = processor.process_video(
            video_path=video_path,
            text_query=text_query,
            output_path=output_path,
            display=False
        )
        
        print("\nProcessing completed!")
        print(f"Output saved to: {output_path}")
        print(f"Statistics: {stats}")
    else:
        print(f"Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")

if __name__ == "__main__":
    demo_usage()