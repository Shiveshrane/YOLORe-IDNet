"""
CLIP-based Person Identifier for YOLORe-IDNet
Uses CLIP as initial identifier for new entities, then hands off to ReID tracking
"""

import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import logging

class CLIPPersonIdentifier:
    """
    CLIP-based person identifier for initial entity recognition
    Uses CLIP only when new people enter the scene to check if they match target descriptions
    """
    
    def __init__(self, device: str = None, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model for person identification
        
        Args:
            device: Device to run CLIP on ('cuda' or 'cpu')
            model_name: Hugging Face model name for CLIP
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model and processor from transformers
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"CLIP model {model_name} loaded on {self.device}")
        
        # Cache for target descriptions to avoid re-encoding
        self.target_embeddings_cache = {}
    
    def add_target_description(self, target_id: int, description: str):
        """
        Add a target description and cache its embedding with enhanced variations
        
        Args:
            target_id: Unique identifier for the target
            description: Text description of the target
        """
        # Create multiple description variations for better matching
        descriptions = [
            description,
            f"a person {description}",
            f"someone {description}",
            f"a human {description}"
        ]
        
        embeddings = []
        for desc in descriptions:
            # Preprocess and encode text
            inputs = self.processor(text=[desc], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features.cpu())
        
        # Average embeddings for robustness
        avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
        
        self.target_embeddings_cache[target_id] = {
            'description': description,
            'embedding': avg_embedding,
            'matches_found': 0,
            'variations': descriptions
        }
        
        self.logger.info(f"Added target {target_id}: '{description}' with {len(descriptions)} variations")
    
    def remove_target_description(self, target_id: int):
        """Remove a target description from cache"""
        if target_id in self.target_embeddings_cache:
            desc = self.target_embeddings_cache[target_id]['description']
            del self.target_embeddings_cache[target_id]
            self.logger.info(f"Removed target {target_id}: '{desc}'")
    
    def clear_all_targets(self):
        """Clear all target descriptions"""
        count = len(self.target_embeddings_cache)
        self.target_embeddings_cache.clear()
        self.logger.info(f"Cleared {count} target descriptions")
    
    def identify_new_person(self, frame: np.ndarray, person_bbox: List[int], 
                           confidence_threshold: float = 0.25) -> Optional[Tuple[int, float, str]]:
        """
        Identify if a new person matches any of the target descriptions
        This is the main method called when a new person enters the scene
        
        Args:
            frame: Input frame (BGR format)
            person_bbox: Bounding box [x1, y1, x2, y2] of the detected person
            confidence_threshold: Minimum similarity score to consider a match
            
        Returns:
            Tuple of (target_id, similarity_score, description) if match found, None otherwise
        """
        if not self.target_embeddings_cache:
            return None
        
        # Extract person crop
        person_crop = self._extract_person_crop(frame, person_bbox)
        if person_crop is None:
            return None
        
        # Encode the person image
        person_embedding = self._encode_person_image(person_crop)
        if person_embedding is None:
            return None
        
        # Compare with all target descriptions
        best_match = None
        best_score = confidence_threshold
        
        for target_id, target_data in self.target_embeddings_cache.items():
            target_embedding = target_data['embedding'].to(self.device)
            
            # Compute similarity
            similarity = torch.cosine_similarity(person_embedding, target_embedding).item()
            
            if similarity > best_score:
                best_score = similarity
                best_match = (target_id, similarity, target_data['description'])
        
        if best_match:
            target_id = best_match[0]
            self.target_embeddings_cache[target_id]['matches_found'] += 1
            self.logger.info(f"New person matches target {target_id}: {best_score:.3f}")
        
        return best_match
    
    def _extract_person_crop(self, frame: np.ndarray, bbox: List[int]) -> Optional[Image.Image]:
        """Extract person crop from frame given bounding box with enhanced preprocessing"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Add padding to include more context - improves CLIP performance
            padding = 0.15  # 15% padding
            bbox_w, bbox_h = x2 - x1, y2 - y1
            pad_x = int(bbox_w * padding)
            pad_y = int(bbox_h * padding)
            
            # Apply padding with bounds checking
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w, x2 + pad_x)
            y2_pad = min(h, y2 + pad_y)
            
            # Extract person crop with padding
            person_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if person_crop.size > 0:
                # Convert BGR to RGB and create PIL Image
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(person_crop_rgb)
                
                # Resize to optimal size for CLIP (improves performance)
                pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
                
                return pil_image
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting person crop: {e}")
            return None
    
    def _encode_person_image(self, person_image: Image.Image) -> Optional[torch.Tensor]:
        """Encode a person image using CLIP"""
        try:
            # Preprocess image
            inputs = self.processor(images=person_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features
            
        except Exception as e:
            self.logger.error(f"Error encoding person image: {e}")
            return None
    
    def get_target_statistics(self) -> Dict:
        """Get statistics about target matching"""
        stats = {}
        for target_id, target_data in self.target_embeddings_cache.items():
            stats[target_id] = {
                'description': target_data['description'],
                'matches_found': target_data['matches_found']
            }
        return stats

# Integration helper functions for new entity identification workflow
def identify_new_entities(yolo_detections: List, frame: np.ndarray, 
                         clip_identifier: CLIPPersonIdentifier,
                         existing_tracked_persons: List[Dict]) -> List[Dict]:
    """
    Identify new entities that match target descriptions
    This is called when new people are detected to check if they match any targets
    
    Args:
        yolo_detections: YOLO detection results
        frame: Current frame
        clip_identifier: CLIPPersonIdentifier instance
        existing_tracked_persons: List of already tracked persons (to avoid duplicates)
        
    Returns:
        List of new entities that match target descriptions
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
        return []
    
    new_entities = []
    
    # Check each detected person against target descriptions
    for i, detection in enumerate(person_detections):
        bbox = detection['bbox']
        
        # Skip if this detection overlaps significantly with existing tracked persons
        if _overlaps_with_existing(bbox, existing_tracked_persons):
            continue
        
        # Use CLIP to identify if this new person matches any target
        match_result = clip_identifier.identify_new_person(frame, bbox)
        
        if match_result:
            target_id, similarity_score, description = match_result
            
            new_entities.append({
                'detection_index': i,
                'detection': detection,
                'target_id': target_id,
                'similarity_score': similarity_score,
                'description': description,
                'status': 'newly_identified'
            })
    
    return new_entities

def _overlaps_with_existing(bbox: List[float], existing_tracked_persons: List[Dict], 
                           iou_threshold: float = 0.3) -> bool:
    """
    Check if a bounding box overlaps significantly with existing tracked persons
    
    Args:
        bbox: Bounding box to check [x1, y1, x2, y2]
        existing_tracked_persons: List of tracked persons with bboxes
        iou_threshold: IoU threshold for overlap detection
        
    Returns:
        True if bbox overlaps with any existing person
    """
    x1, y1, x2, y2 = bbox
    
    for person in existing_tracked_persons:
        if 'bbox' not in person:
            continue
            
        px1, py1, px2, py2 = person['bbox']
        
        # Calculate IoU
        intersection_x1 = max(x1, px1)
        intersection_y1 = max(y1, py1)
        intersection_x2 = min(x2, px2)
        intersection_y2 = min(y2, py2)
        
        if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
            bbox_area = (x2 - x1) * (y2 - y1)
            person_area = (px2 - px1) * (py2 - py1)
            union_area = bbox_area + person_area - intersection_area
            
            iou = intersection_area / union_area if union_area > 0 else 0
            
            if iou > iou_threshold:
                return True
    
    return False