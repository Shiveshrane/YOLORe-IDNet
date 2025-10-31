"""
Enhanced server.py with CLIP integration for text-based person selection
"""

from flask import Flask, request, jsonify
import torchvision.ops.boxes as bops
import base64
import cv2
import torch
import numpy as np
import pandas as pd
import time
import sys
from Alignedreid_demo import Aligned_Reid_class
from clip_person_selector import CLIPPersonIdentifier, identify_new_entities
from PIL import Image
import logging
import json

# Initializations
app = Flask(__name__)
tracker = cv2.legacy.TrackerKCF_create()
ReId = Aligned_Reid_class()

# CLIP Integration - for initial entity identification only
clip_identifier = CLIPPersonIdentifier()

# Global variables
init_bb = None
suspect_features = None
flag = False
frame_data = None
has_exit_count = 0
currently_selected_src = None
has_suspect_exit = False
exit_iou_vector = []
Reid_flag = False
position = None
camera_id = None
num_of_suspect_features = None
first_call = True

# Multi-target text-based selection variables
tracked_targets = {}  # Dictionary to store multiple targets: {target_id: {query, features, bbox, etc.}}
clip_selection_mode = False
next_target_id = 1

# Load the YOLOv5 model
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if sys.modules.get('models') is not None:
    sys.modules.pop('models')

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.to(device)

def torch_normalizer(tensor):
    normalized_tensor = tensor / tensor.norm(dim=1, keepdim=True)
    return normalized_tensor

def decode_base64_image(encoded_string):
    """Decode base64 encoded image"""
    try:
        img_data = base64.b64decode(encoded_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return None

def encode_image_to_base64(image):
    """Encode image to base64"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode()
        return img_str
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return None

@app.route('/add_target', methods=['POST'])
def add_target():
    """
    Add a new target with text query for CLIP-based selection
    """
    global tracked_targets, clip_selection_mode, next_target_id
    
    try:
        data = request.get_json()
        text_query = data.get('text_query', '')
        target_name = data.get('target_name', f'Target_{next_target_id}')
        
        if not text_query.strip():
            return jsonify({
                'status': 'error',
                'message': 'Empty text query provided'
            }), 400
        
        target_id = next_target_id
        tracked_targets[target_id] = {
            'name': target_name,
            'text_query': text_query.strip(),
            'features': None,
            'bbox': None,
            'last_seen': None,
            'status': 'searching',
            'confidence': 0.0
        }
        
        next_target_id += 1
        clip_selection_mode = True
        
        # Add target description to CLIP identifier
        clip_identifier.add_target_description(target_id, text_query.strip())
        
        logging.info(f"Added target {target_id}: {target_name} - '{text_query}'")
        
        return jsonify({
            'status': 'success',
            'target_id': target_id,
            'target_name': target_name,
            'message': f'Target added: {target_name}',
            'total_targets': len(tracked_targets)
        })
        
    except Exception as e:
        logging.error(f"Error adding target: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/remove_target', methods=['POST'])
def remove_target():
    """
    Remove a target from tracking
    """
    global tracked_targets, clip_selection_mode
    
    try:
        data = request.get_json()
        target_id = data.get('target_id')
        
        if target_id is None:
            return jsonify({
                'status': 'error',
                'message': 'Target ID required'
            }), 400
        
        if target_id in tracked_targets:
            target_name = tracked_targets[target_id]['name']
            
            # Remove from CLIP identifier
            clip_identifier.remove_target_description(target_id)
            
            del tracked_targets[target_id]
            
            # Disable CLIP mode if no targets remain
            if not tracked_targets:
                clip_selection_mode = False
            
            logging.info(f"Removed target {target_id}: {target_name}")
            
            return jsonify({
                'status': 'success',
                'message': f'Target {target_name} removed',
                'total_targets': len(tracked_targets)
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Target not found'
            }), 404
        
    except Exception as e:
        logging.error(f"Error removing target: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/get_targets', methods=['GET'])
def get_targets():
    """
    Get list of all tracked targets
    """
    targets_info = []
    
    for target_id, target_data in tracked_targets.items():
        targets_info.append({
            'target_id': target_id,
            'name': target_data['name'],
            'text_query': target_data['text_query'],
            'status': target_data['status'],
            'confidence': target_data['confidence'],
            'last_seen': target_data['last_seen']
        })
    
    return jsonify({
        'targets': targets_info,
        'total_targets': len(tracked_targets),
        'clip_mode': clip_selection_mode
    })

@app.route('/clear_all_targets', methods=['POST'])
def clear_all_targets():
    """
    Clear all tracked targets
    """
    global tracked_targets, clip_selection_mode
    
    targets_count = len(tracked_targets)
    
    # Clear from CLIP identifier
    clip_identifier.clear_all_targets()
    
    tracked_targets.clear()
    clip_selection_mode = False
    
    return jsonify({
        'status': 'success',
        'message': f'Cleared {targets_count} targets',
        'total_targets': 0
    })

@app.route('/disable_clip_mode', methods=['POST'])
def disable_clip_mode():
    """
    Disable CLIP mode and clear all targets
    """
    global clip_selection_mode, tracked_targets
    
    clip_selection_mode = False
    targets_count = len(tracked_targets)
    
    # Clear from CLIP identifier
    clip_identifier.clear_all_targets()
    
    tracked_targets.clear()
    
    return jsonify({
        'status': 'success',
        'message': f'CLIP mode disabled, cleared {targets_count} targets',
        'clip_mode': False,
        'total_targets': 0
    })

@app.route('/get_clip_status', methods=['GET'])
def get_clip_status():
    """
    Get current CLIP mode status and targets
    """
    return jsonify({
        'clip_mode': clip_selection_mode,
        'total_targets': len(tracked_targets),
        'targets': list(tracked_targets.keys()),
        'has_active_targets': any(target['status'] == 'tracking' for target in tracked_targets.values())
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Enhanced prediction endpoint with multi-target CLIP integration
    """
    global init_bb, suspect_features, flag, frame_data, has_exit_count
    global currently_selected_src, has_suspect_exit, exit_iou_vector
    global Reid_flag, position, camera_id, num_of_suspect_features, first_call
    global tracked_targets, clip_selection_mode
    
    try:
        data = request.get_json()
        
        # Decode the main image
        encoded_img = data.get('image')
        if not encoded_img:
            return jsonify({'error': 'No image provided'}), 400
        
        frame = decode_base64_image(encoded_img)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run YOLO detection
        results = model(frame)
        detections = results.pandas().xyxy[0].values.tolist()
        
        # Filter for person detections
        person_detections = [det for det in detections if det[5] == 0]  # class 0 = person
        
        response_data = {}
        
        # New Entity Identification with CLIP (only for initial detection)
        if clip_selection_mode and tracked_targets and person_detections:
            try:
                # Get list of currently tracked persons to avoid duplicates
                currently_tracked = [target for target in tracked_targets.values() 
                                   if target['status'] == 'tracking']
                
                # Use CLIP to identify new entities that match target descriptions
                new_entities = identify_new_entities(
                    detections, frame, clip_identifier, currently_tracked
                )
                
                matched_targets = []
                
                # Process newly identified entities
                for entity in new_entities:
                    target_id = entity['target_id']
                    detection = entity['detection']
                    similarity_score = entity['similarity_score']
                    description = entity['description']
                    
                    # Extract person crop for ReID feature extraction
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size > 0:
                        # Extract ReID features for future tracking
                        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(person_crop_rgb)
                        features = ReId.get_features(pil_image)
                        
                        # Update target data - transition from searching to tracking
                        tracked_targets[target_id].update({
                            'features': features,
                            'bbox': bbox,
                            'last_seen': time.time(),
                            'status': 'tracking',  # Now using ReID for tracking
                            'confidence': similarity_score,
                            'initial_clip_score': similarity_score
                        })
                        
                        matched_targets.append({
                            'target_id': target_id,
                            'target_name': tracked_targets[target_id]['name'],
                            'bbox': bbox,
                            'confidence': detection['confidence'],
                            'similarity_score': similarity_score,
                            'text_query': description,
                            'detection_index': entity['detection_index'],
                            'tracking_method': 'clip_initial'
                        })
                        
                        logging.info(f"CLIP identified new entity for target {target_id}: {similarity_score:.3f}")
                
                # Continue tracking existing targets using ReID only
                for target_id, target_data in tracked_targets.items():
                    if target_data['status'] == 'tracking' and target_id not in [e['target_id'] for e in new_entities]:
                        # Use ReID to track this person
                        target_features = target_data['features']
                        best_match_idx = None
                        best_reid_score = 0.0
                        
                        # Find best ReID match among remaining detections
                        for i, det in enumerate(person_detections):
                            # Skip if this detection was already claimed by a new entity
                            if any(e['detection_index'] == i for e in new_entities):
                                continue
                                
                            bbox = det[:4]
                            x1, y1, x2, y2 = map(int, bbox)
                            person_crop = frame[y1:y2, x1:x2]
                            
                            if person_crop.size > 0:
                                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                                pil_image = Image.fromarray(person_crop_rgb)
                                current_features = ReId.get_features(pil_image)
                                
                                # Compute ReID similarity
                                reid_score = torch.cosine_similarity(
                                    target_features.unsqueeze(0), 
                                    current_features.unsqueeze(0)
                                ).item()
                                
                                if reid_score > best_reid_score and reid_score > 0.5:  # ReID threshold
                                    best_reid_score = reid_score
                                    best_match_idx = i
                        
                        if best_match_idx is not None:
                            # Update target with new detection
                            det = person_detections[best_match_idx]
                            bbox = det[:4]
                            
                            tracked_targets[target_id].update({
                                'bbox': bbox,
                                'last_seen': time.time(),
                                'confidence': best_reid_score
                            })
                            
                            matched_targets.append({
                                'target_id': target_id,
                                'target_name': target_data['name'],
                                'bbox': bbox,
                                'confidence': det[4],
                                'reid_score': best_reid_score,
                                'text_query': target_data['text_query'],
                                'detection_index': best_match_idx,
                                'tracking_method': 'reid_tracking'
                            })
                            
                            logging.info(f"ReID tracking target {target_id}: {best_reid_score:.3f}")
                        else:
                            # Target lost - mark for potential re-identification
                            if time.time() - target_data['last_seen'] > 3.0:  # 3 seconds timeout
                                tracked_targets[target_id]['status'] = 'searching'  # Allow CLIP re-identification
                                logging.warning(f"Target {target_id} lost, switching back to CLIP search")
                
                # Prepare response
                response_data = {
                    'entity_identification_tracking': True,
                    'matched_targets': matched_targets,
                    'total_detections': len(person_detections),
                    'new_entities_found': len(new_entities),
                    'active_targets': len([t for t in tracked_targets.values() if t['status'] in ['searching', 'tracking']]),
                    'ReidStatus': len(matched_targets) > 0,
                    'message': f'Tracking {len(matched_targets)} targets ({len(new_entities)} newly identified via CLIP)'
                }
                
                # Set ReID flag if any targets are being tracked
                Reid_flag = any(target['status'] == 'tracking' for target in tracked_targets.values())
                    
            except Exception as e:
                logging.error(f"Entity identification error: {e}")
                response_data = {
                    'entity_identification_tracking': False,
                    'error': f'Entity identification failed: {str(e)}',
                    'ReidStatus': False
                }
        
        # Regular ReID mode (existing functionality)
        elif 'suspect_img' in data and 'suspect_feat_img' in data:
            # Handle existing ReID pipeline
            suspect_img_encoded = data.get('suspect_img')
            suspect_feat_img_encoded = data.get('suspect_feat_img')
            
            if suspect_img_encoded and suspect_feat_img_encoded:
                suspect_img = decode_base64_image(suspect_img_encoded)
                suspect_feat_img = decode_base64_image(suspect_feat_img_encoded)
                
                if suspect_img is not None and suspect_feat_img is not None:
                    # Process with existing ReID pipeline
                    # ... (existing ReID logic here)
                    pass
        
        # Standard detection mode
        else:
            # Return all person detections for manual selection
            detection_results = []
            for i, det in enumerate(person_detections):
                detection_results.append({
                    'id': i,
                    'bbox': det[:4],
                    'confidence': det[4],
                    'class': 'person'
                })
            
            response_data = {
                'detections': detection_results,
                'total_detections': len(person_detections),
                'clip_mode': clip_selection_mode,
                'total_targets': len(tracked_targets),
                'ReidStatus': False
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'ReidStatus': False
        }), 500

@app.route('/clip_preview', methods=['POST'])
def clip_preview():
    """
    Preview which new entities would be identified by current target descriptions
    """
    try:
        data = request.get_json()
        encoded_img = data.get('image')
        
        if not encoded_img:
            return jsonify({'error': 'Image required'}), 400
        
        frame = decode_base64_image(encoded_img)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run YOLO detection
        results = model(frame)
        detections = results.pandas().xyxy[0].values.tolist()
        person_detections = [det for det in detections if det[5] == 0]
        
        if not person_detections:
            return jsonify({
                'preview_results': [],
                'message': 'No persons detected'
            })
        
        # Get potential matches for all current targets
        preview_results = []
        
        for i, detection in enumerate(person_detections):
            bbox = detection[:4]
            
            # Check against all target descriptions
            match_result = clip_identifier.identify_new_person(frame, bbox, confidence_threshold=0.15)
            
            if match_result:
                target_id, similarity_score, description = match_result
                target_name = tracked_targets.get(target_id, {}).get('name', f'Target_{target_id}')
                
                preview_results.append({
                    'detection_index': i,
                    'bbox': bbox,
                    'confidence': detection[4],
                    'target_id': target_id,
                    'target_name': target_name,
                    'similarity_score': similarity_score,
                    'description': description
                })
        
        return jsonify({
            'preview_results': preview_results,
            'total_persons': len(person_detections),
            'potential_matches': len(preview_results),
            'message': f'Found {len(preview_results)} potential matches from {len(person_detections)} detected persons'
        })
        
    except Exception as e:
        logging.error(f"Preview error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Starting YOLORe-IDNet server with Multi-Target CLIP integration...")
    print("Multi-Target CLIP mode endpoints:")
    print("  POST /add_target - Add a new target with text query")
    print("  POST /remove_target - Remove a specific target")
    print("  GET /get_targets - Get list of all tracked targets")
    print("  POST /clear_all_targets - Clear all tracked targets")
    print("  POST /disable_clip_mode - Disable CLIP mode")
    print("  GET /get_clip_status - Get CLIP mode status")
    print("  POST /clip_preview - Preview CLIP matches")
    print("  POST /predict - Main prediction endpoint (enhanced with multi-target CLIP)")
    print("\nFeatures:")
    print("  - Multi-target text-based person tracking")
    print("  - CLIP + ReID hybrid tracking")
    print("  - Automatic target association and handoff")
    
    app.run(host='0.0.0.0', port=5000, debug=True)