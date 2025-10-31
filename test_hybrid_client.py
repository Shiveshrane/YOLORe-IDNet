"""
Simple Client for Testing Hybrid Multi-Camera Tracking Server
"""

import cv2
import numpy as np
import requests
import base64
import json
import time
import argparse
from typing import Dict, List

class HybridTrackingClient:
    """Client for testing the hybrid tracking server"""
    
    def __init__(self, server_url: str = 'http://127.0.0.1:5001'):
        self.server_url = server_url
        self.session = requests.Session()
        print(f"🔗 Connected to server: {server_url}")
    
    def encode_frame(self, frame):
        """Encode frame to base64"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def make_request(self, endpoint: str, method: str = 'GET', data: dict = None):
        """Make API request"""
        url = f"{self.server_url}{endpoint}"
        
        try:
            if method == 'POST':
                response = self.session.post(url, json=data, timeout=10)
            else:
                response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: {e}")
            return None
    
    def setup_system(self, camera_configs: List[dict], target_descriptions: List[dict]):
        """Setup cameras and targets"""
        print("🚀 Setting up hybrid tracking system...")
        
        # Clear existing setup
        result = self.make_request('/clear_all', 'POST')
        if result:
            print("🧹 Cleared existing configuration")
        
        # Add cameras
        print("\n📹 Adding cameras...")
        for config in camera_configs:
            result = self.make_request('/add_camera', 'POST', {
                'camera_id': config['id'],
                'tracker_type': config.get('tracker_type', 'deepsort')
            })
            
            if result:
                print(f"  ✅ {config['id']} ({config.get('tracker_type', 'deepsort')})")
            else:
                print(f"  ❌ Failed to add {config['id']}")
        
        # Add targets
        print("\n🎯 Adding target descriptions...")
        target_ids = []
        for target in target_descriptions:
            result = self.make_request('/add_target', 'POST', {
                'target_name': target['name'],
                'description': target['description']
            })
            
            if result:
                target_id = result['target_id']
                target_ids.append(target_id)
                print(f"  ✅ Target {target_id}: {target['description']}")
            else:
                print(f"  ❌ Failed to add target: {target['description']}")
        
        return target_ids
    
    def simulate_camera_views(self, frame, camera_configs):
        """Simulate multiple camera views by cropping"""
        camera_frames = {}
        
        for config in camera_configs:
            camera_id = config['id']
            crop = config.get('crop', (0, 0, frame.shape[1], frame.shape[0]))
            x1, y1, x2, y2 = crop
            
            # Ensure crop is within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                cropped = frame[y1:y2, x1:x2]
                # Resize for consistency
                cropped = cv2.resize(cropped, (320, 240))
                camera_frames[camera_id] = cropped
        
        return camera_frames
    
    def visualize_results(self, camera_frames, tracking_results):
        """Visualize tracking results"""
        # Colors for different targets
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        visualized_frames = {}
        
        for camera_id, frame in camera_frames.items():
            vis_frame = frame.copy()
            
            # Get tracking results for this camera
            camera_result = tracking_results.get('per_camera_results', {}).get(camera_id, {})
            tracks = camera_result.get('tracks', [])
            
            for track in tracks:
                bbox = track['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Choose color based on global target ID
                global_target_id = track.get('global_target_id')
                if global_target_id is not None:
                    color = colors[global_target_id % len(colors)]
                    label = f"T{global_target_id}"
                    if 'target_name' in track:
                        label += f": {track['target_name']}"
                else:
                    color = (128, 128, 128)  # Gray for unidentified
                    label = f"Track {track['local_track_id']}"
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(vis_frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
                cv2.putText(vis_frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Draw confidence
                conf_text = f"{track['confidence']:.2f}"
                cv2.putText(vis_frame, conf_text, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add camera label
            cv2.putText(vis_frame, camera_id.upper(), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            visualized_frames[camera_id] = vis_frame
        
        return visualized_frames
    
    def create_dashboard(self, visualized_frames, tracking_results):
        """Create dashboard view"""
        if len(visualized_frames) >= 4:
            # 2x2 grid
            camera_ids = list(visualized_frames.keys())[:4]
            h, w = next(iter(visualized_frames.values())).shape[:2]
            
            dashboard = np.zeros((h*2 + 20, w*2 + 20, 3), dtype=np.uint8)
            positions = [(0, 0), (w+10, 0), (0, h+10), (w+10, h+10)]
            
            for i, camera_id in enumerate(camera_ids):
                if i < len(positions):
                    x, y = positions[i]
                    dashboard[y:y+h, x:x+w] = visualized_frames[camera_id]
            
            # Add statistics
            summary = tracking_results.get('summary', {})
            stats_text = f"Targets: {summary.get('total_targets', 0)} | Tracks: {summary.get('total_tracks', 0)}"
            cv2.putText(dashboard, stats_text, (10, dashboard.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return dashboard
        else:
            return next(iter(visualized_frames.values()))
    
    def run_demo(self, video_source=0, max_frames=1000):
        """Run tracking demo"""
        print(f"🎬 Starting demo with video source: {video_source}")
        
        # Open video source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ Could not open video source: {video_source}")
            return
        
        # Camera configurations for simulation
        camera_configs = [
            {'id': 'camera_1', 'tracker_type': 'deepsort', 'crop': (0, 0, 320, 240)},
            {'id': 'camera_2', 'tracker_type': 'deepsort', 'crop': (320, 0, 640, 240)},
            {'id': 'camera_3', 'tracker_type': 'deepsort', 'crop': (0, 240, 320, 480)},
            {'id': 'camera_4', 'tracker_type': 'deepsort', 'crop': (320, 240, 640, 480)}
        ]
        
        # Target descriptions
        target_descriptions = [
            {'name': 'Person in Red', 'description': 'person wearing red clothing'},
            {'name': 'Person with Bag', 'description': 'person carrying a bag or backpack'},
            {'name': 'Walking Person', 'description': 'person walking or moving'}
        ]
        
        # Setup system
        target_ids = self.setup_system(camera_configs, target_descriptions)
        
        if not target_ids:
            print("❌ Failed to setup system")
            return
        
        frame_count = 0
        processing_times = []
        
        print("\n🎬 Starting video processing... Press 'q' to quit")
        
        try:
            while frame_count < max_frames:
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video")
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Simulate multi-camera views
                camera_frames = self.simulate_camera_views(frame, camera_configs)
                
                # Encode frames
                encoded_frames = {}
                for camera_id, cam_frame in camera_frames.items():
                    encoded_frames[camera_id] = self.encode_frame(cam_frame)
                
                # Send tracking request
                tracking_results = self.make_request('/predict_hybrid', 'POST', {
                    'camera_frames': encoded_frames
                })
                
                if tracking_results:
                    # Visualize results
                    visualized_frames = self.visualize_results(camera_frames, tracking_results)
                    dashboard = self.create_dashboard(visualized_frames, tracking_results)
                    
                    # Display
                    cv2.imshow('Hybrid Multi-Camera Tracking Demo', dashboard)
                    
                    # Print periodic updates
                    if frame_count % 30 == 0:
                        global_targets = tracking_results.get('global_targets', [])
                        print(f"Frame {frame_count:4d} | Targets: {len(global_targets):2d} | " +
                              f"FPS: {1.0/(time.time()-start_time):.1f}")
                        
                        for target in global_targets[:3]:  # Show first 3 targets
                            cameras = target.get('active_cameras', [])
                            conf = target.get('confidence', 0)
                            print(f"  • {target['name']}: {cameras} (conf: {conf:.3f})")
                
                # Record processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                frame_count += 1
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⚡ Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            if processing_times:
                avg_time = np.mean(processing_times)
                avg_fps = 1.0 / avg_time
                print(f"\n📊 Performance Statistics:")
                print(f"  Frames processed: {frame_count}")
                print(f"  Average processing time: {avg_time:.3f}s")
                print(f"  Average FPS: {avg_fps:.1f}")
            
            # Get final status
            status = self.make_request('/get_status')
            if status:
                print(f"\n📋 Final Status:")
                print(f"  Active cameras: {len(status['active_cameras'])}")
                print(f"  Total targets: {status['total_targets']}")

def main():
    parser = argparse.ArgumentParser(description='Test Hybrid Multi-Camera Tracking System')
    parser.add_argument('--server', default='http://127.0.0.1:5001', help='Server URL')
    parser.add_argument('--video', default=0, help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--frames', type=int, default=1000, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Convert video argument
    video_source = args.video
    if video_source != '0' and not video_source.isdigit():
        # It's a file path
        pass
    else:
        # It's a camera index
        video_source = int(video_source)
    
    print("🚀 Hybrid Multi-Camera Tracking Client")
    print("=" * 50)
    print(f"Server: {args.server}")
    print(f"Video source: {video_source}")
    print(f"Max frames: {args.frames}")
    print()
    
    # Create client and run demo
    client = HybridTrackingClient(args.server)
    client.run_demo(video_source, args.frames)
    
    print("\n✅ Demo completed!")

if __name__ == '__main__':
    main()
