"""
Setup and Testing Script for Hybrid Multi-Camera Tracking System
"""

import subprocess
import sys
import os
import time
import platform
import argparse
from pathlib import Path

def run_command(cmd, check=True, capture_output=False):
    """Run shell command"""
    print(f"🔧 Running: {cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False

def check_python():
    """Check Python installation"""
    print("🐍 Checking Python installation...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ required")
        return False
    
    print("✅ Python version OK")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    # Base requirements
    base_packages = [
        "opencv-python",
        "torch",
        "torchvision", 
        "numpy",
        "Pillow",
        "flask",
        "requests",
        "matplotlib",
        "scikit-learn",
        "scipy"
    ]
    
    # Advanced tracking requirements (Kalman + Hungarian)
    advanced_packages = [
        "filterpy",  # Kalman Filter
        "scipy"      # Hungarian Algorithm (linear_sum_assignment)
    ]
    
    # YOLOv5 requirements
    yolo_packages = [
        "ultralytics",
        "pyyaml",
        "tqdm"
    ]
    
    # CLIP requirements  
    clip_packages = [
        "transformers",
        "ftfy",
        "regex"
    ]
    
    all_packages = base_packages + advanced_packages + yolo_packages + clip_packages
    
    for package in all_packages:
        print(f"📦 Installing {package}...")
        if not run_command(f"pip install {package}"):
            print(f"⚠️  Warning: Failed to install {package}")
    
    print("✅ Package installation completed")

def download_models():
    """Download required model files"""
    print("\n🔽 Downloading models...")
    
    # Check if YOLOv5 model exists
    yolo_model = "yolov5n.pt"
    if not os.path.exists(yolo_model):
        print(f"📥 Downloading {yolo_model}...")
        try:
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            print(f"✅ {yolo_model} ready")
        except Exception as e:
            print(f"⚠️  Warning: Could not download YOLOv5 model: {e}")
    else:
        print(f"✅ {yolo_model} already exists")
    
    print("✅ Model download completed")

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    dirs = [
        "log",
        "aligned",
        "models", 
        "util",
        "evaluation_scripts",
        "__pycache__"
    ]
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"📁 Created: {dir_name}")
        else:
            print(f"✅ Exists: {dir_name}")
    
    print("✅ Directory setup completed")

def check_files():
    """Check if required files exist"""
    print("\n📋 Checking required files...")
    
    required_files = [
        "hybrid_server.py",
        "test_hybrid_client.py", 
        "hybrid_tracking_test.ipynb",
        "main_with_clip.py",
        "clip_person_selector.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("   Please ensure all required files are in the current directory")
        return False
    
    print("✅ All required files found")
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    
    test_imports = [
        ("cv2", "opencv-python"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("flask", "flask"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib")
    ]
    
    # Test advanced tracking imports
    advanced_imports = [
        ("filterpy.kalman", "filterpy"),
        ("scipy.optimize", "scipy")
    ]
    
    failed_imports = []
    for module, package in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} (install: pip install {package})")
            failed_imports.append((module, package))
    
    # Test advanced tracking capabilities
    print("\n🧪 Testing advanced tracking imports...")
    for module, package in advanced_imports:
        try:
            __import__(module)
            print(f"✅ {module} (Kalman+Hungarian available)")
        except ImportError:
            print(f"❌ {module} (install: pip install {package})")
            failed_imports.append((module, package))
    
    if failed_imports:
        print("\n⚠️  Failed imports found. Run:")
        for module, package in failed_imports:
            print(f"   pip install {package}")
        return False
    
    print("✅ All imports successful")
    return True

def start_server(background=True):
    """Start the hybrid server"""
    print("\n🚀 Starting hybrid server...")
    
    if not os.path.exists("hybrid_server.py"):
        print("❌ hybrid_server.py not found")
        return False
    
    try:
        if background:
            if platform.system() == "Windows":
                # Windows
                subprocess.Popen([sys.executable, "hybrid_server.py"], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # Unix-like
                subprocess.Popen([sys.executable, "hybrid_server.py"])
            
            print("✅ Server started in background")
            print("   Check console for server logs")
            
            # Wait a moment for server to start
            time.sleep(3)
            return True
        else:
            # Run in foreground
            run_command(f"{sys.executable} hybrid_server.py", check=False)
            return True
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def test_server():
    """Test if server is running"""
    print("\n🔍 Testing server connection...")
    
    try:
        import requests
        response = requests.get("http://127.0.0.1:5001/get_status", timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding")
            status = response.json()
            print(f"   Cameras: {len(status.get('active_cameras', []))}")
            print(f"   Targets: {status.get('total_targets', 0)}")
            return True
        else:
            print(f"❌ Server returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def run_client_test():
    """Run client test"""
    print("\n🎬 Starting client test...")
    
    if not os.path.exists("test_hybrid_client.py"):
        print("❌ test_hybrid_client.py not found")
        return False
    
    print("Starting client demo...")
    print("Press 'q' in the video window to quit")
    
    try:
        run_command(f"{sys.executable} test_hybrid_client.py --frames 100", check=False)
        return True
    except Exception as e:
        print(f"❌ Client test failed: {e}")
        return False

def open_notebook():
    """Open Jupyter notebook"""
    print("\n📓 Opening Jupyter notebook...")
    
    if not os.path.exists("hybrid_tracking_test.ipynb"):
        print("❌ hybrid_tracking_test.ipynb not found")
        return False
    
    try:
        run_command("jupyter notebook hybrid_tracking_test.ipynb", check=False)
        return True
    except Exception as e:
        print(f"❌ Failed to open notebook: {e}")
        print("   Try: pip install jupyter")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup Hybrid Multi-Camera Tracking System')
    parser.add_argument('--install', action='store_true', help='Install requirements')
    parser.add_argument('--setup', action='store_true', help='Full setup (install + download models)')
    parser.add_argument('--server', action='store_true', help='Start server only')
    parser.add_argument('--test', action='store_true', help='Run client test')
    parser.add_argument('--notebook', action='store_true', help='Open Jupyter notebook')
    parser.add_argument('--full', action='store_true', help='Full setup and test')
    
    args = parser.parse_args()
    
    print("🚀 Hybrid Multi-Camera Tracking Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python():
        sys.exit(1)
    
    if args.install or args.setup or args.full:
        install_requirements()
        
    if args.setup or args.full:
        setup_directories()
        download_models()
    
    # Always check files and imports
    if not check_files():
        print("\n❌ Setup incomplete - missing files")
        sys.exit(1)
        
    if not test_imports():
        print("\n❌ Setup incomplete - missing dependencies")
        sys.exit(1)
    
    if args.server or args.full:
        if start_server():
            time.sleep(2)
            test_server()
        
    if args.test or args.full:
        if test_server():
            run_client_test()
        else:
            print("❌ Cannot test - server not running")
    
    if args.notebook:
        open_notebook()
    
    if not any([args.install, args.setup, args.server, args.test, args.notebook, args.full]):
        # Default behavior - show help
        print("\n📖 Usage Examples:")
        print("  Full setup:      python setup_hybrid.py --full")
        print("  Install only:    python setup_hybrid.py --install") 
        print("  Start server:    python setup_hybrid.py --server")
        print("  Run test:        python setup_hybrid.py --test")
        print("  Open notebook:   python setup_hybrid.py --notebook")
        print("\n🎯 Quick Start:")
        print("  1. python setup_hybrid.py --setup")
        print("  2. python setup_hybrid.py --server")
        print("  3. python setup_hybrid.py --test")
        
    print("\n✅ Setup script completed!")

if __name__ == '__main__':
    main()
