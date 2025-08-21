# CLIP Integration for YOLORe-IDNet

This branch adds **CLIP (Contrastive Language-Image Pre-training)** integration to YOLORe-IDNet, enabling **text-based person selection** instead of manual bounding box assignment.

## 🚀 New Features

### Text-Based Person Selection
- **Natural Language Queries**: Describe the target person in plain English
- **Automatic Selection**: CLIP automatically identifies the best matching person
- **Similarity Scoring**: Get confidence scores for person matches
- **Multi-Match Preview**: See multiple potential matches ranked by similarity

### Enhanced User Interface
- **CLIP Control Panel**: Dedicated UI for text-based selection
- **Real-time Preview**: Preview CLIP matches before selection
- **Confidence Threshold**: Adjustable similarity thresholds
- **Status Indicators**: Clear feedback on CLIP mode status

## 📁 New Files

### Core CLIP Integration
- `clip_person_selector.py` - Main CLIP integration module
- `server_with_clip.py` - Enhanced server with CLIP endpoints
- `main_with_clip.py` - Enhanced client with CLIP UI
- `requirements_with_clip.txt` - Updated dependencies

## 🛠️ Installation

### 1. Install Dependencies
```bash
pip install -r requirements_with_clip.txt
```

### 2. Install CLIP
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 3. Verify Installation
```python
import clip
import torch

# Test CLIP installation
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP installed successfully!")
```

## 🚀 Quick Start

### 1. Start the Enhanced Server
```bash
python server_with_clip.py
```

### 2. Start the Enhanced Client
```bash
python main_with_clip.py
```

### 3. Use Text-Based Selection
1. Enable "CLIP Mode" in the control panel
2. Enter a description like: `"person wearing red shirt and blue jeans"`
3. Click "Set Query" or press Enter
4. The system automatically selects the best matching person

## 📖 Usage Examples

### Text Query Examples
```python
# Clothing-based descriptions
"person wearing red shirt"
"woman in blue dress"
"man with black jacket"

# Physical characteristics
"tall person with dark hair"
"person with glasses"
"woman with long blonde hair"

# Combined descriptions
"person wearing red shirt and blue jeans"
"tall man in white t-shirt and black pants"
"woman with glasses wearing green jacket"
```

### API Usage

#### Set Text Query
```python
import requests

response = requests.post('http://localhost:5000/set_text_query', 
                        json={'text_query': 'person wearing red shirt'})
```

#### Preview Matches
```python
response = requests.post('http://localhost:5000/clip_preview', 
                        json={
                            'image': base64_encoded_image,
                            'text_query': 'person wearing red shirt'
                        })
```

#### Get CLIP Status
```python
response = requests.get('http://localhost:5000/get_clip_status')
status = response.json()
print(f"CLIP Mode: {status['clip_mode']}")
print(f"Current Query: {status['text_query']}")
```

## 🔧 Configuration

### CLIP Model Selection
```python
# In clip_person_selector.py
# Available models: "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"
model, preprocess = clip.load("ViT-B/32", device=device)
```

### Confidence Thresholds
```python
# Adjust in CLIPPersonSelector
confidence_threshold = 0.2  # Default threshold
top_k = 3  # Number of top matches to consider
```

### Performance Optimization
```python
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Batch processing for multiple detections
batch_size = 8  # Adjust based on GPU memory
```

## 🏗️ Architecture

### CLIP Integration Flow
```
1. YOLO Detection → Person Bounding Boxes
2. Crop Extraction → Individual Person Images  
3. CLIP Encoding → Image & Text Features
4. Similarity Computation → Cosine Similarity Scores
5. Best Match Selection → Highest Scoring Person
6. ReID Feature Extraction → Traditional Pipeline
```

### Class Structure
```python
CLIPPersonSelector
├── encode_text_query()          # Text → Features
├── encode_person_images()       # Images → Features  
├── compute_similarity_scores()  # Cosine Similarity
├── select_person_by_text()      # Best Match Selection
└── get_multiple_matches()       # Top-K Matches
```

## 🔄 Integration with Existing Pipeline

### Backward Compatibility
- **Manual Selection**: Original bounding box selection still works
- **Dual Mode**: Switch between CLIP and manual modes seamlessly
- **Existing ReID**: CLIP selection feeds into existing ReID pipeline

### API Endpoints
```
POST /predict              # Enhanced with CLIP support
POST /set_text_query       # Set CLIP text query
POST /disable_clip_mode    # Disable CLIP mode
GET  /get_clip_status      # Get CLIP status
POST /clip_preview         # Preview CLIP matches
```

## 🎯 Performance Considerations

### Speed Optimization
- **Model Caching**: CLIP model loaded once at startup
- **Batch Processing**: Process multiple detections together
- **GPU Acceleration**: Automatic GPU usage when available

### Memory Management
- **Feature Normalization**: L2 normalized features for efficiency
- **Tensor Cleanup**: Proper memory cleanup after processing
- **Device Management**: Automatic device selection

### Accuracy Tuning
- **Threshold Adjustment**: Tune confidence thresholds per use case
- **Query Optimization**: Use descriptive, specific text queries
- **Multi-Match Review**: Check top-K matches for better selection

## 🐛 Troubleshooting

### Common Issues

#### CLIP Installation
```bash
# If CLIP installation fails
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

#### CUDA Issues
```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### Memory Issues
```python
# Reduce batch size or use CPU
device = "cpu"  # Force CPU usage
batch_size = 1  # Process one detection at a time
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

### Planned Features
- **Multi-Modal Queries**: Combine text + reference images
- **Temporal Consistency**: Track persons across frames using CLIP
- **Custom Fine-tuning**: Domain-specific CLIP model training
- **Advanced Queries**: Support for complex compositional descriptions

### Integration Possibilities
- **Face Recognition**: Combine with face recognition models
- **Pose Estimation**: Include pose-based descriptions
- **Attribute Detection**: Detailed clothing and accessory detection
- **Scene Understanding**: Context-aware person selection

## 📊 Evaluation

### Metrics
- **Selection Accuracy**: Percentage of correct person selections
- **Similarity Scores**: Distribution of CLIP similarity scores
- **Processing Time**: Latency analysis for real-time performance
- **User Satisfaction**: Qualitative feedback on text-based selection

### Benchmarking
```python
# Example evaluation script
from clip_person_selector import CLIPPersonSelector

selector = CLIPPersonSelector()
# Run evaluation on test dataset
# Measure accuracy, speed, and user satisfaction
```

## 🤝 Contributing

### Development Setup
```bash
git checkout clip-integration
pip install -r requirements_with_clip.txt
pip install -e .  # Development installation
```

### Testing
```bash
python -m pytest tests/test_clip_integration.py
```

### Code Style
```bash
black clip_person_selector.py
flake8 clip_person_selector.py
```

## 📄 License

This CLIP integration maintains the same license as the original YOLORe-IDNet project.

## 🙏 Acknowledgments

- **OpenAI CLIP**: For the foundational vision-language model
- **YOLORe-IDNet**: Original multi-camera person tracking system
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library

---

## 📞 Support

For issues specific to CLIP integration:
1. Check this documentation
2. Review the troubleshooting section
3. Open an issue with detailed error logs
4. Include system specifications and CLIP model used

**Happy tracking with natural language! 🎯**