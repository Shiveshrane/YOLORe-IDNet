"""
Enhanced main.py with CLIP integration for text-based person selection
"""

from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, QPushButton, 
                           QDialog, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,
                           QTextEdit, QCheckBox, QGroupBox, QSlider, QSpinBox)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from threading import Thread
import base64
import json
import pandas as pd
import numpy as np
import cv2
import imutils
import time
from datetime import datetime
from collections import deque
import sys
import requests
from multiprocessing import Pool
import numpy as np
import datetime
from recommender import camera_Recommender
from location_extractor import VideoMetadataExtractor
import urllib

# Global variables
cam_recommender = camera_Recommender()
meta_extractor = VideoMetadataExtractor('urls.txt')
target_gps_data = []

class CLIPControlWidget(QWidget):
    """
    Widget for Multi-Target CLIP-based text selection controls
    """
    targets_updated = pyqtSignal()
    clip_mode_toggled = pyqtSignal(bool)
    preview_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.server_url = 'http://127.0.0.1:5000'
        self.tracked_targets = {}
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Multi-Target CLIP Mode Group
        clip_group = QGroupBox("Multi-Target CLIP Tracking")
        clip_layout = QVBoxLayout()
        
        # Enable/Disable CLIP
        self.clip_enabled = QCheckBox("Enable Multi-Target CLIP Mode")
        self.clip_enabled.toggled.connect(self.on_clip_mode_toggled)
        clip_layout.addWidget(self.clip_enabled)
        
        # Add new target section
        add_target_group = QGroupBox("Add New Target")
        add_layout = QVBoxLayout()
        
        # Target name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Target Name:"))
        self.target_name_input = QLineEdit()
        self.target_name_input.setPlaceholderText("e.g., 'Person A', 'Suspect 1'")
        name_layout.addWidget(self.target_name_input)
        add_layout.addLayout(name_layout)
        
        # Text query input
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("Description:"))
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("e.g., 'person wearing red shirt and blue jeans'")
        self.text_input.returnPressed.connect(self.on_add_target)
        query_layout.addWidget(self.text_input)
        
        self.add_target_btn = QPushButton("Add Target")
        self.add_target_btn.clicked.connect(self.on_add_target)
        query_layout.addWidget(self.add_target_btn)
        
        add_layout.addLayout(query_layout)
        add_target_group.setLayout(add_layout)
        clip_layout.addWidget(add_target_group)
        
        # Targets list
        targets_group = QGroupBox("Active Targets")
        targets_layout = QVBoxLayout()
        
        # Targets list widget
        self.targets_list = QTextEdit()
        self.targets_list.setMaximumHeight(150)
        self.targets_list.setReadOnly(True)
        targets_layout.addWidget(self.targets_list)
        
        # Target management buttons
        target_buttons_layout = QHBoxLayout()
        
        self.refresh_targets_btn = QPushButton("Refresh Targets")
        self.refresh_targets_btn.clicked.connect(self.refresh_targets)
        target_buttons_layout.addWidget(self.refresh_targets_btn)
        
        self.clear_all_btn = QPushButton("Clear All Targets")
        self.clear_all_btn.clicked.connect(self.clear_all_targets)
        target_buttons_layout.addWidget(self.clear_all_btn)
        
        targets_layout.addLayout(target_buttons_layout)
        targets_group.setLayout(targets_layout)
        clip_layout.addWidget(targets_group)
        
        # Preview button
        self.preview_btn = QPushButton("Preview All Matches")
        self.preview_btn.clicked.connect(self.on_preview_matches)
        clip_layout.addWidget(self.preview_btn)
        
        # Status display
        self.status_label = QLabel("Multi-Target CLIP Mode: Disabled")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        clip_layout.addWidget(self.status_label)
        
        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("CLIP Confidence Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(10, 50)
        self.threshold_slider.setValue(20)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("0.20")
        threshold_layout.addWidget(self.threshold_label)
        
        clip_layout.addLayout(threshold_layout)
        
        clip_group.setLayout(clip_layout)
        layout.addWidget(clip_group)
        
        # Instructions
        instructions = QTextEdit()
        instructions.setMaximumHeight(120)
        instructions.setPlainText(
            "Entity Identification & Tracking Instructions:\n"
            "1. Enable Multi-Target CLIP mode\n"
            "2. Add targets with descriptions (e.g., 'woman in red dress')\n"
            "3. CLIP identifies new entities matching descriptions\n"
            "4. System automatically switches to ReID for tracking\n"
            "5. CLIP re-activates if targets are lost\n"
            "6. View real-time status of all targets"
        )
        instructions.setReadOnly(True)
        layout.addWidget(instructions)
        
        self.setLayout(layout)
        
        # Initially disable controls
        self.set_controls_enabled(False)
    
    def set_controls_enabled(self, enabled):
        """Enable/disable CLIP controls"""
        self.target_name_input.setEnabled(enabled)
        self.text_input.setEnabled(enabled)
        self.add_target_btn.setEnabled(enabled)
        self.preview_btn.setEnabled(enabled)
        self.threshold_slider.setEnabled(enabled)
        self.refresh_targets_btn.setEnabled(enabled)
        self.clear_all_btn.setEnabled(enabled)
    
    def on_clip_mode_toggled(self, checked):
        """Handle CLIP mode toggle"""
        self.set_controls_enabled(checked)
        
        if checked:
            self.status_label.setText("Multi-Target CLIP Mode: Enabled")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.refresh_targets()
        else:
            self.status_label.setText("Multi-Target CLIP Mode: Disabled")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.disable_clip_mode()
        
        self.clip_mode_toggled.emit(checked)
    
    def on_threshold_changed(self, value):
        """Handle threshold slider changes"""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
    
    def on_add_target(self):
        """Add a new target"""
        target_name = self.target_name_input.text().strip()
        text_query = self.text_input.text().strip()
        
        if not target_name:
            target_name = f"Target_{len(self.tracked_targets) + 1}"
        
        if not text_query:
            self.status_label.setText("Error: Empty description")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            return
        
        try:
            response = requests.post(f"{self.server_url}/add_target", 
                                   json={
                                       'target_name': target_name,
                                       'text_query': text_query
                                   })
            
            if response.status_code == 200:
                data = response.json()
                target_id = data['target_id']
                
                self.tracked_targets[target_id] = {
                    'name': target_name,
                    'query': text_query,
                    'status': 'searching'
                }
                
                self.status_label.setText(f"Added target: {target_name}")
                self.status_label.setStyleSheet("color: blue; font-weight: bold;")
                
                # Clear inputs
                self.target_name_input.clear()
                self.text_input.clear()
                
                # Refresh display
                self.refresh_targets()
                
            else:
                self.status_label.setText("Error adding target")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                
        except Exception as e:
            self.status_label.setText(f"Connection error: {str(e)[:20]}...")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def refresh_targets(self):
        """Refresh targets list from server"""
        try:
            response = requests.get(f"{self.server_url}/get_targets")
            
            if response.status_code == 200:
                data = response.json()
                targets = data['targets']
                
                # Update local tracking
                self.tracked_targets.clear()
                for target in targets:
                    self.tracked_targets[target['target_id']] = {
                        'name': target['name'],
                        'query': target['text_query'],
                        'status': target['status'],
                        'confidence': target.get('confidence', 0.0)
                    }
                
                # Update display
                self.update_targets_display()
                
                self.targets_updated.emit()
                
        except Exception as e:
            print(f"Error refreshing targets: {e}")
    
    def update_targets_display(self):
        """Update the targets list display"""
        if not self.tracked_targets:
            self.targets_list.setPlainText("No targets added yet.")
            return
        
        display_text = ""
        for target_id, target_data in self.tracked_targets.items():
            status_color = {
                'searching': '🔍',
                'tracking': '🎯',
                'lost': '❌'
            }.get(target_data['status'], '❓')
            
            confidence = target_data.get('confidence', 0.0)
            confidence_str = f" ({confidence:.2f})" if confidence > 0 else ""
            
            display_text += f"{status_color} {target_data['name']}: {target_data['query'][:40]}{'...' if len(target_data['query']) > 40 else ''}{confidence_str}\n"
        
        self.targets_list.setPlainText(display_text.strip())
    
    def clear_all_targets(self):
        """Clear all targets"""
        try:
            response = requests.post(f"{self.server_url}/clear_all_targets")
            
            if response.status_code == 200:
                self.tracked_targets.clear()
                self.update_targets_display()
                self.status_label.setText("All targets cleared")
                self.status_label.setStyleSheet("color: blue; font-weight: bold;")
                
        except Exception as e:
            self.status_label.setText(f"Error clearing targets: {str(e)[:20]}...")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def on_preview_matches(self):
        """Request preview of matches"""
        self.preview_requested.emit()
    
    def disable_clip_mode(self):
        """Disable CLIP mode on server"""
        try:
            response = requests.post(f"{self.server_url}/disable_clip_mode")
            if response.status_code == 200:
                self.tracked_targets.clear()
                self.update_targets_display()
        except:
            pass

class EnhancedYOLO:
    """Enhanced YOLO class with CLIP entity identification"""
    
    def __init__(self, url='http://127.0.0.1:5000/predict'):
        self.url = url
        self.encoded_img = None
        self.do_ReID = False
        self.camId = None
        self.new_bounding_box_cord = None
        self.length = None
        self.recommender = camera_Recommender()
        self.session = requests.Session()
        self.current_cam_id = None
        self.image = None
        self.clip_mode = False
    
    def set_clip_mode(self, enabled):
        """Set CLIP mode state (simplified since targets are managed server-side)"""
        self.clip_mode = enabled
    
    def scale_bbox(self, bbox, original_size, resized_size):
        """Scale bounding box coordinates"""
        x, y, w, h = bbox
        original_height, original_width = original_size
        resized_height, resized_width = resized_size

        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        return x, y, w, h
    
    def predict(self, suspect_img_encoded=None, suspect_feature1_encoded=None):
        """Enhanced prediction with entity identification support"""
        if self.encoded_img is None:
            return None, None
        
        try:
            # Prepare request data
            request_data = {'image': self.encoded_img}
            
            # Add suspect images if provided (for traditional ReID)
            if suspect_img_encoded and suspect_feature1_encoded:
                request_data.update({
                    'suspect_img': suspect_img_encoded,
                    'suspect_feat_img': suspect_feature1_encoded
                })
            
            # Send request
            json_response = self.session.post(self.url, json=request_data)
            json_data = json_response.json()
            
            # Handle entity identification tracking response
            if json_data.get('entity_identification_tracking'):
                self.do_ReID = json_data.get('ReidStatus', False)
                return 'entity_identification_tracking', json_data
            
            # Handle regular detection response
            elif 'detections' in json_data:
                detections = json_data['detections']
                self.do_ReID = json_data.get('ReidStatus', False)
                
                if detections:
                    # Convert to pandas DataFrame format for compatibility
                    results = []
                    for det in detections:
                        bbox = det['bbox']
                        results.append([
                            bbox[0], bbox[1], bbox[2], bbox[3],  # x1, y1, x2, y2
                            det['confidence'], 0, 'person'  # conf, class, name
                        ])
                    
                    results = np.array(results)
                    return 'show_box', results
            
            # Handle ReID status
            elif json_data.get('ReidStatus'):
                self.do_ReID = True
                return 'reid', None
            
            # Handle errors
            elif 'error' in json_data:
                print(f"Server error: {json_data['error']}")
                return 'error', json_data['error']
            
            return None, None
            
        except Exception as e:
            print(f'Exception in predict: {e}')
            return 'error', str(e)

class EnhancedCameraWidget(QWidget):
    """Enhanced camera widget with CLIP integration"""
    
    frame_fetch = False
    encoded_frames_for_reid = {}
    active_cam = None
    suspect_img_encoded = None
    suspect_feature1_encoded = None
    neighbours = None
    trajectory = []
    
    def __init__(self, width, height, stream_link=0, id=None, aspect_ratio=False, 
                 parent=None, deque_size=1):
        super(EnhancedCameraWidget, self).__init__(parent)
        
        # Initialize attributes
        self.results_path = '/results/'
        self.suspect_data = []
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.suspect_img_encoded = None
        self.suspect_feature1_encoded = None
        self.deque = deque(maxlen=deque_size)
        self.arr = []
        self.frame = None
        self.start_time = 0
        self.end_time = 0
        self._fps_data = []
        self.url = 'http://127.0.0.1:5000/predict'
        self.yolo = EnhancedYOLO()
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio
        self.cv_frame = None
        self.frame_id = id
        self.start_detection = False
        self.camera_stream_link = stream_link
        self.online = False
        self.capture = None
        
        # Multi-target CLIP-related attributes
        self.clip_mode = False
        self.tracked_targets = {}
        self.last_detections = []
        
        # UI setup
        self.video_frame = QtWidgets.QLabel()
        self.load_network_stream()
        
        # Start background frame grabbing
        self.get_frame_thread = Thread(target=self.get_frame, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()
        
        # Timer for frame updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_frame)
        self.timer.start()
        
        print(f'Started camera: {self.camera_stream_link}')
    
    def set_clip_mode(self, enabled, targets=None):
        """Set CLIP mode for this camera"""
        self.clip_mode = enabled
        if targets:
            self.tracked_targets = targets
        else:
            self.tracked_targets = {}
        
        self.yolo.set_clip_mode(enabled)
        
        if enabled:
            print(f"Multi-target CLIP mode enabled for camera {self.frame_id}: {len(self.tracked_targets)} targets")
        else:
            print(f"CLIP mode disabled for camera {self.frame_id}")
    
    def load_network_stream(self):
        """Load network stream"""
        def load_network_stream_thread():
            if self.verify_network_stream(self.camera_stream_link):
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True
        
        self.load_stream_thread = Thread(target=load_network_stream_thread, args=())
        self.load_stream_thread.daemon = True
        self.load_stream_thread.start()
    
    def verify_network_stream(self, link):
        """Verify if network stream is accessible"""
        try:
            cap = cv2.VideoCapture(link)
            if not cap.isOpened():
                return False
            cap.release()
            return True
        except:
            return False
    
    def get_frame(self):
        """Get frame from camera stream"""
        while True:
            try:
                if self.online:
                    ret, frame = self.capture.read()
                    if ret:
                        self.deque.append(frame)
                    else:
                        print(f"Failed to read frame from camera {self.frame_id}")
                        self.online = False
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in get_frame: {e}")
                time.sleep(0.1)
    
    def set_frame(self):
        """Set frame for display and processing"""
        if not self.online or len(self.deque) == 0:
            return
        
        try:
            frame = self.deque[-1]
            self.cv_frame = frame.copy()
            
            # Encode frame for YOLO processing
            _, buffer = cv2.imencode('.jpg', frame)
            self.yolo.encoded_img = base64.b64encode(buffer).decode()
            self.yolo.current_cam_id = self.frame_id
            
            # Process with YOLO (including multi-target CLIP if enabled)
            if self.start_detection:
                status, data = self.yolo.predict(
                    self.suspect_img_encoded, 
                    self.suspect_feature1_encoded
                )
                
                if status == 'entity_identification_tracking':
                    self.handle_entity_identification_tracking(data)
                elif status == 'show_box':
                    self.draw_detections(frame, data)
                elif status == 'reid':
                    self.handle_reid_mode(frame)
                elif status == 'error':
                    print(f"Detection error: {data}")
                
                self.last_detections = data if isinstance(data, list) else []
            
            # Display frame
            self.display_frame(frame)
            
        except Exception as e:
            print(f"Error in set_frame: {e}")
    
    def handle_entity_identification_tracking(self, data):
        """Handle entity identification and tracking results"""
        if not data or not data.get('matched_targets'):
            return
        
        matched_targets = data['matched_targets']
        new_entities = data.get('new_entities_found', 0)
        total_targets = data.get('active_targets', 0)
        
        print(f"Camera {self.frame_id}: Tracking {len(matched_targets)}/{total_targets} targets ({new_entities} newly identified)")
        
        # Draw all matched targets on frame
        if self.cv_frame is not None:
            # Add header info
            header_text = f"Entity Tracking: {len(matched_targets)}/{total_targets} targets"
            if new_entities > 0:
                header_text += f" ({new_entities} new via CLIP)"
            cv2.putText(self.cv_frame, header_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Colors for different targets
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 255, 128), (255, 128, 128)]
            
            for i, target in enumerate(matched_targets):
                bbox = target['bbox']
                target_name = target['target_name']
                target_id = target['target_id']
                tracking_method = target.get('tracking_method', 'unknown')
                
                # Choose color based on tracking method
                if tracking_method == 'clip_initial':
                    color = (0, 255, 0)  # Green for newly identified via CLIP
                elif tracking_method == 'reid_tracking':
                    color = (255, 0, 0)  # Blue for ReID tracking
                else:
                    color = colors[i % len(colors)]
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(self.cv_frame, (x1, y1), (x2, y2), color, 3)
                
                # Prepare label based on tracking method
                if tracking_method == 'clip_initial':
                    score = target.get('similarity_score', 0.0)
                    label = f"{target_name} (CLIP: {score:.2f})"
                elif tracking_method == 'reid_tracking':
                    score = target.get('reid_score', 0.0)
                    label = f"{target_name} (ReID: {score:.2f})"
                else:
                    confidence = target.get('confidence', 0.0)
                    label = f"{target_name} ({confidence:.2f})"
                
                # Label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(self.cv_frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
                
                # Label text
                cv2.putText(self.cv_frame, label, (x1, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Add status indicator
                status_text = "NEW" if tracking_method == 'clip_initial' else "TRACKING"
                cv2.putText(self.cv_frame, status_text, (x1, y2+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def handle_clip_selection(self, data):
        """Handle CLIP-based person selection (legacy - kept for compatibility)"""
        bbox = data['bbox']
        similarity_score = data['similarity_score']
        text_query = data['text_query']
        
        print(f"CLIP selected person: {text_query} (score: {similarity_score:.3f})")
        
        # Draw selection on frame
        if self.cv_frame is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(self.cv_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add label
            label = f"CLIP: {similarity_score:.3f}"
            cv2.putText(self.cv_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add query text
            cv2.putText(self.cv_frame, text_query[:30], (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes on frame"""
        for detection in detections:
            x1, y1, x2, y2, conf = detection[:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add confidence label
            label = f"Person: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    def handle_reid_mode(self, frame):
        """Handle ReID tracking mode"""
        # Add ReID tracking visualization
        cv2.putText(frame, "Multi-Target ReID Tracking Active", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def display_frame(self, frame):
        """Display frame in Qt widget"""
        try:
            # Resize frame to fit widget
            if self.maintain_aspect_ratio:
                frame = imutils.resize(frame, width=self.screen_width)
            else:
                frame = cv2.resize(frame, (self.screen_width, self.screen_height))
            
            # Convert to Qt format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, 
                                  QtGui.QImage.Format_RGB888)
            
            # Display
            self.video_frame.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            
        except Exception as e:
            print(f"Error displaying frame: {e}")

class MainWindow(QMainWindow):
    """Enhanced main window with CLIP controls"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLORe-IDNet with CLIP Integration")
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Camera display area
        camera_layout = QVBoxLayout()
        
        # Create camera widgets (example with 4 cameras)
        self.cameras = []
        camera_grid = QtWidgets.QGridLayout()
        
        # Example camera streams (replace with actual URLs)
        camera_urls = [
            0,  # Webcam
            # Add more camera URLs here
        ]
        
        for i, url in enumerate(camera_urls):
            camera = EnhancedCameraWidget(
                width=640, height=480, 
                stream_link=url, id=i
            )
            self.cameras.append(camera)
            camera_grid.addWidget(camera.video_frame, i//2, i%2)
        
        camera_layout.addLayout(camera_grid)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self.stop_detection)
        control_layout.addWidget(self.stop_btn)
        
        camera_layout.addLayout(control_layout)
        main_layout.addLayout(camera_layout)
        
        # CLIP control panel
        self.clip_controls = CLIPControlWidget()
        self.clip_controls.clip_mode_toggled.connect(self.on_clip_mode_toggled)
        self.clip_controls.text_query_changed.connect(self.on_text_query_changed)
        self.clip_controls.preview_requested.connect(self.on_preview_requested)
        
        main_layout.addWidget(self.clip_controls)
        
        central_widget.setLayout(main_layout)
    
    def start_detection(self):
        """Start detection on all cameras"""
        for camera in self.cameras:
            camera.start_detection = True
        print("Detection started")
    
    def stop_detection(self):
        """Stop detection on all cameras"""
        for camera in self.cameras:
            camera.start_detection = False
        print("Detection stopped")
    
    def on_clip_mode_toggled(self, enabled):
        """Handle CLIP mode toggle"""
        text_query = self.clip_controls.text_input.text().strip()
        
        for camera in self.cameras:
            camera.set_clip_mode(enabled, text_query if enabled else None)
    
    def on_text_query_changed(self, text):
        """Handle text query changes"""
        if self.clip_controls.clip_enabled.isChecked():
            for camera in self.cameras:
                camera.current_text_query = text
    
    def on_preview_requested(self):
        """Handle preview request"""
        # Implementation for preview functionality
        print("Preview requested - implement preview logic here")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    print("YOLORe-IDNet with CLIP Integration started")
    print("Features:")
    print("- Text-based person selection using CLIP")
    print("- Traditional manual bounding box selection")
    print("- Multi-camera ReID tracking")
    print("- Real-time preview of CLIP matches")
    
    sys.exit(app.exec_())