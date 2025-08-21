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
from clip_person_selector import CLIPPersonSelector, integrate_clip_with_yolo_reid
from PIL import Image
import logging
import json

# Initializations
app = Flask(__name__)
tracker = cv2.legacy.TrackerKCF_create()
ReId = Aligned_Reid_class()

# CLIP Integration
clip_selector = CLIPPersonSelector()

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

# Text-based selection variables
current_text_query = None
clip_selection_mode = False
selected_person_features = None

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

@app.route('/set_text_query', methods=['POST'])
def set_text_query():
    """
    Set text query for CLIP-based person selection
    """
    global current_text_query, clip_selection_mode, selected_person_features
    
    try:
        data = request.get_json()
        text_query = data.get('text_query', '')
        
        if not text_query.strip():
            return jsonify({
                'status': 'error',
                'message': 'Empty text query provided'
            }), 400
        
        current_text_query = text_query.strip()
        clip_selection_mode = True
        selected_person_features = None
        
        logging.info(f"Text query set: {current_text_query}")
        
        return jsonify({
            'status': 'success',
            'message': f'Text query set: {current_text_query}',
            'clip_mode': True
        })
        
    except Exception as e:
        logging.error(f"Error setting text query: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/disable_clip_mode', methods=['POST'])
def disable_clip_mode():
    """
    Disable CLIP mode and return to manual selection
    """
    global clip_selection_mode, current_text_query, selected_person_features
    
    clip_selection_mode = False
    current_text_query = None
    selected_person_features = None
    
    return jsonify({
        'status': 'success',
        'message': 'CLIP mode disabled',
        'clip_mode': False
    })

@app.route('/get_clip_status', methods=['GET'])
def get_clip_status():
    """
    Get current CLIP mode status
    """
    return jsonify({
        'clip_mode': clip_selection_mode,
        'text_query': current_text_query,
        'has_selected_person': selected_person_features is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Enhanced prediction endpoint with CLIP integration
    """
    global init_bb, suspect_features, flag, frame_data, has_exit_count
    global currently_selected_src, has_suspect_exit, exit_iou_vector
    global Reid_flag, position, camera_id, num_of_suspect_features, first_call
    global current_text_query, clip_selection_mode, selected_person_features
    
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
        
        # CLIP-based selection mode
        if clip_selection_mode and current_text_query and person_detections:
            try:
                # Use CLIP to select person based on text query
                clip_result = integrate_clip_with_yolo_reid(
                    person_detections, frame, current_text_query, clip_selector
                )
                
                if clip_result:
                    detection = clip_result['detection']
                    similarity_score = clip_result['similarity_score']
                    
                    # Extract person crop for ReID feature extraction
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size > 0:
                        # Extract ReID features
                        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(person_crop_rgb)
                        
                        # Get ReID features using existing pipeline
                        features = ReId.get_features(pil_image)
                        selected_person_features = features
                        
                        # Prepare response
                        response_data = {
                            'clip_selection': True,
                            'selected_person': {
                                'bbox': bbox,
                                'confidence': detection['confidence'],
                                'similarity_score': similarity_score,
                                'text_query': current_text_query
                            },
                            'total_detections': len(person_detections),
                            'ReidStatus': True,
                            'message': f'Person selected via CLIP: \"{current_text_query}\" (similarity: {similarity_score:.3f})'
                        }
                        
                        # Set up for ReID tracking
                        init_bb = tuple(bbox)
                        suspect_features = features
                        Reid_flag = True
                        
                        logging.info(f"CLIP selected person with similarity {similarity_score:.3f}")
                    else:
                        response_data = {
                            'clip_selection': False,
                            'error': 'Invalid person crop extracted',
                            'ReidStatus': False
                        }
                else:
                    # No good match found
                    response_data = {
                        'clip_selection': False,
                        'message': f'No person matching \"{current_text_query}\" found',
                        'total_detections': len(person_detections),
                        'ReidStatus': False
                    }
                    
            except Exception as e:
                logging.error(f"CLIP selection error: {e}")
                response_data = {
                    'clip_selection': False,
                    'error': f'CLIP processing failed: {str(e)}',
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
                'text_query': current_text_query,
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
    Preview CLIP matches without selecting
    """
    try:
        data = request.get_json()
        encoded_img = data.get('image')
        text_query = data.get('text_query', current_text_query)
        
        if not encoded_img or not text_query:
            return jsonify({'error': 'Image and text query required'}), 400
        
        frame = decode_base64_image(encoded_img)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run YOLO detection
        results = model(frame)
        detections = results.pandas().xyxy[0].values.tolist()
        person_detections = [det for det in detections if det[5] == 0]
        
        if not person_detections:
            return jsonify({
                'matches': [],
                'message': 'No persons detected'
            })
        
        # Get multiple matches
        person_det_dicts = []
        for det in person_detections:
            person_det_dicts.append({
                'bbox': det[:4],
                'confidence': det[4],
                'class': det[5]
            })
        
        matches = clip_selector.get_multiple_matches(
            frame, person_det_dicts, text_query, top_k=5
        )
        
        # Prepare response
        match_results = []
        for det_idx, similarity_score, detection in matches:
            match_results.append({
                'detection_index': det_idx,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'similarity_score': similarity_score
            })
        
        # Create visualization
        vis_frame = clip_selector.visualize_matches(frame, matches, text_query)
        vis_encoded = encode_image_to_base64(vis_frame)
        
        return jsonify({
            'matches': match_results,
            'total_persons': len(person_detections),
            'visualization': vis_encoded,
            'text_query': text_query
        })
        
    except Exception as e:
        logging.error(f"CLIP preview error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Starting YOLORe-IDNet server with CLIP integration...")
    print("CLIP mode endpoints:")
    print("  POST /set_text_query - Set text query for person selection")
    print("  POST /disable_clip_mode - Disable CLIP mode")
    print("  GET /get_clip_status - Get CLIP mode status")
    print("  POST /clip_preview - Preview CLIP matches")
    print("  POST /predict - Main prediction endpoint (enhanced with CLIP)")
    
    app.run(host='0.0.0.0', port=5000, debug=True)