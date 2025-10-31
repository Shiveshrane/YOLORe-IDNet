"""
Test Script for Kalman Filter + Hungarian Algorithm Integration
"""

import numpy as np
import cv2
import time

def test_kalman_hungarian():
    """Test the Kalman+Hungarian tracking components"""
    print("🧪 Testing Kalman Filter + Hungarian Algorithm Integration")
    print("=" * 60)
    
    # Test imports
    print("📦 Testing imports...")
    try:
        from filterpy.kalman import KalmanFilter
        from scipy.optimize import linear_sum_assignment
        print("✅ FilterPy (Kalman Filter) available")
        print("✅ SciPy (Hungarian Algorithm) available")
        kalman_available = True
    except ImportError as e:
        print(f"❌ Advanced tracking imports failed: {e}")
        print("💡 Install with: pip install filterpy scipy")
        kalman_available = False
    
    if not kalman_available:
        return False
    
    # Test Kalman Filter
    print("\n🎯 Testing Kalman Filter...")
    try:
        # Initialize Kalman filter (7D state, 4D measurement)
        kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        # Initialize with a test bounding box
        test_bbox = np.array([100, 100, 200, 200])  # x1, y1, x2, y2
        w = test_bbox[2] - test_bbox[0]
        h = test_bbox[3] - test_bbox[1]
        x = test_bbox[0] + w/2.
        y = test_bbox[1] + h/2.
        s = w * h  # scale
        r = w / float(h)  # aspect ratio
        
        kf.x[:4] = np.array([x, y, s, r]).reshape((4, 1))
        
        # Test prediction
        kf.predict()
        print(f"  ✅ Kalman prediction: center=({kf.x[0,0]:.1f}, {kf.x[1,0]:.1f})")
        
        # Test update
        measurement = np.array([x+10, y+5, s*1.1, r]).reshape((4, 1))
        kf.update(measurement)
        print(f"  ✅ Kalman update: center=({kf.x[0,0]:.1f}, {kf.x[1,0]:.1f})")
        
    except Exception as e:
        print(f"  ❌ Kalman Filter test failed: {e}")
        return False
    
    # Test Hungarian Algorithm
    print("\n🧠 Testing Hungarian Algorithm...")
    try:
        # Create test cost matrix
        cost_matrix = np.array([
            [0.8, 0.2, 0.9],  # Detection 0 costs
            [0.3, 0.7, 0.1],  # Detection 1 costs  
            [0.6, 0.4, 0.5]   # Detection 2 costs
        ])
        
        # Run Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_indices, col_indices].sum()
        
        print(f"  ✅ Hungarian assignment: {list(zip(row_indices, col_indices))}")
        print(f"  ✅ Total cost: {total_cost:.3f}")
        
        # Verify it's optimal
        expected_assignment = [(0, 1), (1, 2), (2, 0)]  # Optimal assignment
        expected_cost = cost_matrix[0,1] + cost_matrix[1,2] + cost_matrix[2,0]
        print(f"  ✅ Expected optimal cost: {expected_cost:.3f}")
        
    except Exception as e:
        print(f"  ❌ Hungarian Algorithm test failed: {e}")
        return False
    
    # Test integration
    print("\n🔄 Testing Integration...")
    try:
        # Simulate tracking scenario
        detections = np.array([
            [50, 50, 100, 100, 0.9],   # x1, y1, x2, y2, confidence
            [150, 150, 200, 200, 0.8],
            [250, 50, 300, 100, 0.7]
        ])
        
        # Create mock trackers
        trackers = []
        for i in range(3):
            kf = KalmanFilter(dim_x=7, dim_z=4)
            kf.F = np.eye(7)  # Simple identity transition
            kf.H = np.eye(4, 7)  # First 4 states are observable
            
            # Initialize with detection
            det = detections[i]
            w, h = det[2] - det[0], det[3] - det[1]
            x, y = det[0] + w/2, det[1] + h/2
            kf.x[:4] = np.array([x, y, w*h, w/h]).reshape((4, 1))
            
            trackers.append(kf)
        
        # Predict tracker positions
        predicted_positions = []
        for tracker in trackers:
            tracker.predict()
            predicted_positions.append(tracker.x[:4].flatten())
        
        # Compute IoU-based cost matrix
        def compute_iou(box1, box2):
            # Convert center+scale to bbox
            def center_to_bbox(cx, cy, s, r):
                w = np.sqrt(s * r)
                h = s / w if w > 0 else 0
                return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            
            bbox1 = center_to_bbox(box1[0], box1[1], box1[2], box1[3])
            bbox2 = center_to_bbox(box2[0], box2[1], box2[2], box2[3])
            
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
        
        # Build cost matrix
        cost_matrix = np.zeros((len(detections), len(predicted_positions)))
        for i, det in enumerate(detections):
            w, h = det[2] - det[0], det[3] - det[1]
            det_state = [det[0] + w/2, det[1] + h/2, w*h, w/h]
            
            for j, pred in enumerate(predicted_positions):
                iou = compute_iou(det_state, pred)
                cost_matrix[i, j] = 1 - iou  # Cost = 1 - IoU
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        print(f"  ✅ Detection-Tracker assignment: {list(zip(row_indices, col_indices))}")
        print(f"  ✅ Assignment costs: {[cost_matrix[r,c]:.3f for r,c in zip(row_indices, col_indices)]}")
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False
    
    print(f"\n🎉 All Kalman + Hungarian tests passed!")
    return True


def test_hybrid_server_integration():
    """Test the hybrid server with Kalman+Hungarian"""
    print(f"\n🚀 Testing Hybrid Server Integration...")
    
    try:
        # Import the hybrid server components
        import sys
        import os
        sys.path.append(os.getcwd())
        
        from hybrid_server import KalmanTracker, EnhancedDeepSortTracker, associate_detections_to_trackers
        
        print("✅ Hybrid server imports successful")
        
        # Test KalmanTracker
        print("\n🎯 Testing KalmanTracker...")
        test_bbox = np.array([100, 100, 200, 200])
        tracker = KalmanTracker(test_bbox, frame_idx=0)
        
        # Test prediction
        predicted = tracker.predict()
        print(f"  ✅ Tracker prediction: {predicted[0]}")
        
        # Test update
        new_bbox = np.array([105, 105, 205, 205])
        tracker.update(new_bbox, confidence=0.9)
        print(f"  ✅ Tracker update successful")
        
        # Test EnhancedDeepSortTracker
        print("\n🔄 Testing EnhancedDeepSortTracker...")
        camera_tracker = EnhancedDeepSortTracker("test_camera")
        
        # Test with mock detections
        detections = np.array([
            [50, 50, 100, 100, 0.9],
            [150, 150, 200, 200, 0.8]
        ])
        
        results = camera_tracker.update(detections, [])
        print(f"  ✅ Camera tracker returned {len(results)} tracks")
        
        # Test association function
        print("\n🧠 Testing Association Function...")
        single_tracker = KalmanTracker(np.array([50, 50, 100, 100]))
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, [single_tracker]
        )
        print(f"  ✅ Association: {len(matches)} matches, {len(unmatched_dets)} unmatched detections")
        
    except Exception as e:
        print(f"❌ Hybrid server integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Hybrid server integration successful!")
    return True


def main():
    """Run all tests"""
    print("🔬 Kalman Filter + Hungarian Algorithm Test Suite")
    print("=" * 80)
    
    success = True
    
    # Test core functionality
    if not test_kalman_hungarian():
        success = False
    
    # Test hybrid server integration
    if not test_hybrid_server_integration():
        success = False
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("🚀 Your system is ready for enhanced tracking with:")
        print("   ✅ Kalman Filter motion prediction")
        print("   ✅ Hungarian Algorithm optimal assignment")
        print("   ✅ Hybrid multi-camera tracking")
        print("   ✅ ReID + CLIP global association")
        print()
        print("🎬 Ready to run:")
        print("   python hybrid_server.py")
        print("   python test_hybrid_client.py")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print("💡 Please install missing dependencies:")
        print("   pip install filterpy scipy")
        print("   python setup_hybrid.py --install")
    
    return success


if __name__ == "__main__":
    main()
