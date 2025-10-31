#!/usr/bin/env python3
"""
Test script for CLIP integration in YOLORe-IDNet
Validates CLIP functionality and integration with existing pipeline
"""

import cv2
import numpy as np
import torch
import time
import sys
import os
from PIL import Image
import requests
import base64
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from clip_person_selector import CLIPPersonSelector, integrate_clip_with_yolo_reid
    print("✅ CLIP integration modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CLIP modules: {e}")
    sys.exit(1)

def test_clip_installation():
    """Test if CLIP is properly installed"""
    print("\n🔍 Testing CLIP installation...")
    
    try:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"✅ CLIP model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"❌ CLIP installation test failed: {e}")
        return False

def test_clip_person_selector():
    """Test CLIPPersonSelector class"""
    print("\n🔍 Testing CLIPPersonSelector class...")
    
    try:
        # Initialize selector
        selector = CLIPPersonSelector()
        print("✅ CLIPPersonSelector initialized successfully")
        
        # Test text encoding
        text_query = "person wearing red shirt"
        text_features = selector.encode_text_query(text_query)
        print(f"✅ Text encoding successful. Feature shape: {text_features.shape}")
        
        # Create dummy person images
        dummy_images = []
        for i in range(3):
            # Create random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            pil_img = Image.fromarray(img_array)
            dummy_images.append(pil_img)
        
        # Test image encoding
        image_features = selector.encode_person_images(dummy_images)
        print(f"✅ Image encoding successful. Feature shape: {image_features.shape}")
        
        # Test similarity computation
        similarities = selector.compute_similarity_scores(text_features, image_features)
        print(f"✅ Similarity computation successful. Scores: {similarities}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLIPPersonSelector test failed: {e}")
        return False

def test_yolo_integration():
    """Test integration with YOLO detections"""
    print("\n🔍 Testing YOLO integration...")
    
    try:
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create dummy YOLO detections (format: [x1, y1, x2, y2, conf, class])
        dummy_detections = [
            [100, 100, 200, 300, 0.9, 0],  # person detection 1
            [300, 150, 400, 350, 0.8, 0],  # person detection 2
            [500, 200, 600, 400, 0.7, 0],  # person detection 3
        ]
        
        # Initialize selector
        selector = CLIPPersonSelector()
        
        # Test integration function
        result = integrate_clip_with_yolo_reid(
            dummy_detections, frame, "person wearing blue shirt", selector
        )
        
        if result:
            print("✅ YOLO integration successful")
            print(f"   Selected detection: {result['detection']}")
            print(f"   Similarity score: {result['similarity_score']:.3f}")
            print(f"   Text query: {result['text_query']}")
        else:
            print("⚠️  No person selected (expected with random data)")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO integration test failed: {e}")
        return False

def test_server_endpoints():
    """Test server endpoints (if server is running)"""
    print("\n🔍 Testing server endpoints...")
    
    server_url = "http://127.0.0.1:5000"
    
    try:
        # Test server availability
        response = requests.get(f"{server_url}/get_clip_status", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responsive")
            
            # Test setting text query
            query_response = requests.post(
                f"{server_url}/set_text_query",
                json={"text_query": "person wearing red shirt"},
                timeout=5
            )
            
            if query_response.status_code == 200:
                print("✅ Text query endpoint working")
            else:
                print(f"⚠️  Text query endpoint returned: {query_response.status_code}")
            
            # Test disabling CLIP mode
            disable_response = requests.post(f"{server_url}/disable_clip_mode", timeout=5)
            if disable_response.status_code == 200:
                print("✅ Disable CLIP mode endpoint working")
            
            return True
        else:
            print(f"⚠️  Server responded with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running. Start server_with_clip.py to test endpoints")
        return False
    except Exception as e:
        print(f"❌ Server endpoint test failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality"""
    print("\n🔍 Testing visualization...")
    
    try:
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create dummy matches
        dummy_matches = [
            (0, 0.85, {'bbox': [100, 100, 200, 300]}),
            (1, 0.72, {'bbox': [300, 150, 400, 350]}),
            (2, 0.65, {'bbox': [500, 200, 600, 400]}),
        ]
        
        # Initialize selector
        selector = CLIPPersonSelector()
        
        # Test visualization
        vis_frame = selector.visualize_matches(
            frame, dummy_matches, "person wearing red shirt"
        )
        
        print("✅ Visualization successful")
        print(f"   Original frame shape: {frame.shape}")
        print(f"   Visualized frame shape: {vis_frame.shape}")
        
        # Optionally save visualization for manual inspection
        cv2.imwrite("test_clip_visualization.jpg", vis_frame)
        print("   Visualization saved as 'test_clip_visualization.jpg'")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_performance():
    """Test performance metrics"""
    print("\n🔍 Testing performance...")
    
    try:
        selector = CLIPPersonSelector()
        
        # Test text encoding performance
        text_query = "person wearing red shirt and blue jeans"
        start_time = time.time()
        text_features = selector.encode_text_query(text_query)
        text_time = time.time() - start_time
        print(f"✅ Text encoding time: {text_time:.3f}s")
        
        # Test image encoding performance
        dummy_images = []
        for i in range(5):  # Test with 5 person crops
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            pil_img = Image.fromarray(img_array)
            dummy_images.append(pil_img)
        
        start_time = time.time()
        image_features = selector.encode_person_images(dummy_images)
        image_time = time.time() - start_time
        print(f"✅ Image encoding time (5 images): {image_time:.3f}s")
        
        # Test similarity computation performance
        start_time = time.time()
        similarities = selector.compute_similarity_scores(text_features, image_features)
        similarity_time = time.time() - start_time
        print(f"✅ Similarity computation time: {similarity_time:.3f}s")
        
        total_time = text_time + image_time + similarity_time
        print(f"✅ Total processing time: {total_time:.3f}s")
        
        if total_time < 1.0:
            print("✅ Performance is good for real-time usage")
        else:
            print("⚠️  Performance might be slow for real-time usage")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting CLIP Integration Tests for YOLORe-IDNet")
    print("=" * 60)
    
    tests = [
        ("CLIP Installation", test_clip_installation),
        ("CLIP Person Selector", test_clip_person_selector),
        ("YOLO Integration", test_yolo_integration),
        ("Server Endpoints", test_server_endpoints),
        ("Visualization", test_visualization),
        ("Performance", test_performance),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CLIP integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)