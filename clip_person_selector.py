"""
CLIP-based Person Selector for YOLORe-IDNet
Enables text-based person selection instead of manual bounding box assignment
"""

import torch
import clip
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Optional, Dict
import logging

class CLIPPersonSelector:
    """
    CLIP-based person selector that identifies persons based on textual descriptions
    """
    
    def __init__(self, device: str = None):
        """
        Initialize CLIP model for person selection
        
        Args:
            device: Device to run CLIP on ('cuda' or 'cpu')
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"CLIP model loaded on {self.device}")
    
    def preprocess_detections(self, frame: np.ndarray, detections: List[Dict]) -> List[Image.Image]:
        """
        Extract person crops from frame based on YOLO detections
        
        Args:
            frame: Input frame (BGR format)
            detections: List of detection dictionaries with bbox coordinates
            
        Returns:
            List of PIL Images containing person crops
        """
        person_crops = []
        
        for detection in detections:
            # Extract bounding box coordinates
            if 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
            else:
                # Assume detection format: [x1, y1, x2, y2, conf, class]
                x1, y1, x2, y2 = detection[:4]
            
            # Ensure coordinates are integers and within frame bounds
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Extract person crop
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size > 0:
                # Convert BGR to RGB and create PIL Image
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(person_crop_rgb)
                person_crops.append(pil_image)
            else:
                self.logger.warning(f"Empty crop detected: {x1}, {y1}, {x2}, {y2}")
        
        return person_crops
    
    def encode_text_query(self, text_description: str) -> torch.Tensor:
        """
        Encode text description using CLIP
        
        Args:
            text_description: Natural language description of the person
            
        Returns:
            Encoded text features
        """
        # Tokenize and encode text
        text_tokens = clip.tokenize([text_description]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def encode_person_images(self, person_crops: List[Image.Image]) -> torch.Tensor:
        """
        Encode person images using CLIP
        
        Args:
            person_crops: List of PIL Images containing person crops
            
        Returns:
            Encoded image features
        """
        if not person_crops:
            return torch.empty(0, 512).to(self.device)
        
        # Preprocess images
        image_tensors = torch.stack([self.preprocess(crop) for crop in person_crops]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def compute_similarity_scores(self, text_features: torch.Tensor, 
                                image_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between text and image features
        
        Args:
            text_features: Encoded text features
            image_features: Encoded image features
            
        Returns:
            Similarity scores for each person
        """
        # Compute cosine similarity
        similarity_scores = torch.matmul(text_features, image_features.T)
        return similarity_scores.squeeze()
    
    def select_person_by_text(self, frame: np.ndarray, detections: List[Dict], 
                            text_description: str, 
                            confidence_threshold: float = 0.2) -> Optional[Tuple[int, float, Dict]]:
        """
        Select person from detections based on text description
        
        Args:
            frame: Input frame (BGR format)
            detections: List of person detections
            text_description: Natural language description of target person
            confidence_threshold: Minimum similarity score threshold
            
        Returns:
            Tuple of (detection_index, similarity_score, detection_dict) or None
        """
        if not detections:
            self.logger.warning("No detections provided")
            return None
        
        # Extract person crops
        person_crops = self.preprocess_detections(frame, detections)
        
        if not person_crops:
            self.logger.warning("No valid person crops extracted")
            return None
        
        # Encode text and images
        text_features = self.encode_text_query(text_description)
        image_features = self.encode_person_images(person_crops)
        
        # Compute similarities
        similarity_scores = self.compute_similarity_scores(text_features, image_features)
        
        # Find best match
        best_idx = torch.argmax(similarity_scores).item()
        best_score = similarity_scores[best_idx].item()
        
        self.logger.info(f"Best match: Index {best_idx}, Score: {best_score:.3f}")
        
        if best_score >= confidence_threshold:
            return best_idx, best_score, detections[best_idx]
        else:
            self.logger.warning(f"No match above threshold {confidence_threshold}. Best score: {best_score:.3f}")
            return None
    
    def get_multiple_matches(self, frame: np.ndarray, detections: List[Dict], 
                           text_description: str, 
                           top_k: int = 3,
                           confidence_threshold: float = 0.15) -> List[Tuple[int, float, Dict]]:
        """
        Get multiple person matches ranked by similarity
        
        Args:
            frame: Input frame (BGR format)
            detections: List of person detections
            text_description: Natural language description
            top_k: Number of top matches to return
            confidence_threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (detection_index, similarity_score, detection_dict)
        """
        if not detections:
            return []
        
        # Extract person crops
        person_crops = self.preprocess_detections(frame, detections)
        
        if not person_crops:
            return []
        
        # Encode text and images
        text_features = self.encode_text_query(text_description)
        image_features = self.encode_person_images(person_crops)
        
        # Compute similarities
        similarity_scores = self.compute_similarity_scores(text_features, image_features)
        
        # Get top-k matches
        top_scores, top_indices = torch.topk(similarity_scores, 
                                           min(top_k, len(similarity_scores)))
        
        results = []
        for idx, score in zip(top_indices, top_scores):
            idx_val = idx.item()
            score_val = score.item()
            
            if score_val >= confidence_threshold:
                results.append((idx_val, score_val, detections[idx_val]))
        
        return results
    
    def visualize_matches(self, frame: np.ndarray, matches: List[Tuple[int, float, Dict]], 
                         text_description: str) -> np.ndarray:
        """
        Visualize CLIP matches on the frame
        
        Args:
            frame: Input frame
            matches: List of matches from get_multiple_matches
            text_description: Text query used
            
        Returns:
            Frame with visualized matches
        """
        vis_frame = frame.copy()
        
        # Add text description at top
        cv2.putText(vis_frame, f"Query: {text_description}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (det_idx, score, detection) in enumerate(matches):
            color = colors[i % len(colors)]
            
            # Extract bbox
            if 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
            else:
                x1, y1, x2, y2 = detection[:4]
            
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add score label
            label = f"#{i+1}: {score:.3f}"
            cv2.putText(vis_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame

# Example usage and integration helper functions
def integrate_clip_with_yolo_reid(yolo_detections: List, frame: np.ndarray, 
                                text_query: str, clip_selector: CLIPPersonSelector) -> Optional[Dict]:
    """
    Integration helper to use CLIP with existing YOLO ReID pipeline
    
    Args:
        yolo_detections: YOLO detection results
        frame: Current frame
        text_query: Text description of target person
        clip_selector: CLIPPersonSelector instance
        
    Returns:
        Selected detection dictionary or None
    """
    # Filter for person detections (class 0 in COCO)
    person_detections = []
    for det in yolo_detections:
        if len(det) >= 6 and det[5] == 0:  # person class
            person_detections.append({
                'bbox': det[:4],
                'confidence': det[4],
                'class': det[5]
            })
    
    if not person_detections:
        return None
    
    # Use CLIP to select best matching person
    result = clip_selector.select_person_by_text(frame, person_detections, text_query)
    
    if result:
        det_idx, similarity_score, detection = result
        return {
            'detection': detection,
            'similarity_score': similarity_score,
            'text_query': text_query
        }
    
    return None