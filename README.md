# 🎯 YOLORe-IDNet: Hybrid Person Tracking System

## Enhanced Multi-Camera Person Tracking with Kalman Filter + AlignedReID + Hungarian Algorithm

![plot](/Tables/system-design-1.png)

This repository contains an advanced multi-camera person tracking system that combines state-of-the-art computer vision techniques for robust real-time person detection, tracking, and re-identification.

## 🚀 Key Features

### Core Components
- **🎲 Kalman Filter**: Advanced motion prediction and state estimation
- **🧬 AlignedReID**: Deep learning-based person re-identification features
- **🎭 Hungarian Algorithm**: Optimal detection-to-track association
- **🔤 CLIP Integration**: Natural language target descriptions
- **📹 Multi-Camera Support**: Cross-camera person tracking

### System Improvements
- ✅ **KCF-Free Implementation**: Removed KCF dependencies for better performance
- ✅ **Enhanced Tracking**: Pure Kalman + ReID + Hungarian algorithm approach
- ✅ **Natural Language Interface**: CLIP-based target descriptions
- ✅ **Comprehensive Analytics**: Detailed performance metrics
- ✅ **Easy-to-Use Notebook**: Complete implementation in Jupyter notebook

## 🛠️ Quick Setup

### Option 1: Jupyter Notebook (Recommended)
```bash
# Install dependencies
pip install torch torchvision ultralytics transformers opencv-python scipy filterpy scikit-learn

# Open the main notebook
jupyter notebook hybrid_server_testing.ipynb
```

### Option 2: Original Server-Client Setup
```bash
# Step 1: Download dataset from: https://www.kaggle.com/dsv/6369817
# Step 2: Copy it into Project_Repository
# Step 3: Install dependencies
# Step 4: Fire Server
python3 server.py

# Alternative: Docker setup
docker pull vipin2113106/processing_server:current

# Step 5: Fire client
python3 main.py
```

## 📖 Usage

### Jupyter Notebook Approach
1. Open `hybrid_server_testing.ipynb`
2. Configure your video paths and target descriptions
3. Run cells sequentially for complete tracking pipeline
4. View results with comprehensive analytics

### Traditional Server-Client Approach
Use the original setup as described above for distributed processing.
  
## Try out our MCPT dataset ##
Visit -> https://www.kaggle.com/dsv/6369817
```
@misc{vipin gautam_shitala prasad_sharad sinha_2023,
	title={MCPT-dataset},
	url={https://www.kaggle.com/dsv/6369817},
	DOI={10.34740/KAGGLE/DSV/6369817},
	publisher={Kaggle},
	author={Vipin Gautam and Shitala Prasad and Sharad Sinha},
	year={2023}}
```


## Results ##
![plot](/Tables/challenges-overall.png)
![plot](/Tables/qualitative.png)
![plot](/Tables/tab1.png)
![plot](/Tables/Tab2.png)
![plot](/Tables/Tab3.png)



  


