# Complete Training and Testing Guide

This guide provides step-by-step instructions for setting up, training, and testing the KYC Document Validator.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training the Model](#training-the-model)
5. [Testing the Model](#testing-the-model)
6. [Using the Streamlit App](#using-the-streamlit-app)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. System Requirements
- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: At least 2GB free space

### 2. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Windows:**
- Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH
- Verify installation: `tesseract --version`

### 3. Verify Tesseract Installation
```bash
tesseract --version
# Should output version information
```

---

## Initial Setup

### Step 1: Navigate to Project Directory
```bash
cd "/Users/abhishekmalyan/Aadhar Card and Pancard/KYC-Doc-Validator"
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
```

### Step 3: Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- TensorFlow/Keras
- OpenCV
- Tesseract Python wrapper
- Streamlit
- Albumentations
- And other required packages

### Step 5: Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import pytesseract; print('Tesseract: OK')"
python -c "import streamlit; print('Streamlit: OK')"
```

---

## Dataset Preparation

### Step 1: Understand Dataset Structure

Your dataset should be organized as follows:
```
data/
├── train/
│   ├── aadhaar/     # Real Aadhaar card images
│   ├── pan/         # Real PAN card images
│   ├── fake/        # Fake/synthetic document images
│   └── other/       # Other document types
├── val/
│   ├── aadhaar/
│   ├── pan/
│   ├── fake/
│   └── other/
└── test/
    ├── aadhaar/
    ├── pan/
    ├── fake/
    └── other/
```

### Step 2: Collect Images

**Option A: Use Existing Datasets**
1. Clone the referenced repositories:
   ```bash
   # Magnum-Opus (if available)
   git clone https://github.com/bhargav1000/Magnum-Opus.git temp_magnum
   # Copy relevant images to your data folders
   
   # documentClassification (if available)
   git clone https://github.com/Abdus8Samad/documentClassification.git temp_doc
   # Copy relevant images to your data folders
   ```

**Option B: Create Your Own Dataset**
- Collect real Aadhaar and PAN card images (with permission)
- Create fake documents for training
- Use data augmentation to expand dataset

### Step 3: Image Requirements
- **Format**: JPG, JPEG, or PNG
- **Recommended Size**: At least 300x300 pixels (will be resized to 150x150)
- **Quality**: Clear, readable text
- **Quantity**: 
  - Minimum: 50 images per class per split (total ~600 images)
  - Recommended: 200+ images per class per split (total ~2400 images)
  - For MVP: 100-200 images total is acceptable

### Step 4: Create Synthetic Fake Documents

You can use the following Python script to create synthetic fake documents:

```python
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Create augmentation pipeline for fake documents
transform = A.Compose([
    A.RandomRotate90(p=0.3),
    A.Flip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.2),
    A.ElasticTransform(p=0.2, alpha=50, sigma=5),
    A.GridDistortion(p=0.2),
    A.OpticalDistortion(p=0.2),
    A.ShiftScaleRotate(p=0.3),
    A.CoarseDropout(p=0.2, max_holes=8, max_height=16, max_width=16),
])

# Apply to real images to create variations
# Save to data/train/fake/ or data/val/fake/
```

### Step 5: Verify Dataset Structure
```bash
# Count images in each folder
find data/train -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l
find data/val -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l
find data/test -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l
```

---

## Training the Model

### Step 1: Basic Training Command

```bash
python src/train.py --data_dir data --epochs 10 --batch_size 32
```

### Step 2: Training with Custom Parameters

```bash
python src/train.py \
    --data_dir data \
    --epochs 20 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --model_save_path models/kyc_validator_v1.h5
```

### Step 3: Understanding Training Output

During training, you'll see:
- **Epoch progress**: Loss and accuracy for each epoch
- **Validation metrics**: Performance on validation set
- **Model checkpointing**: Best model saved automatically
- **Early stopping**: Training stops if no improvement

Example output:
```
Epoch 1/10
25/25 [==============================] - 45s 1.8s/step - loss: 1.2345 - ensemble_classification_loss: 0.9876 - final_authenticity_loss: 0.2469 - ensemble_classification_accuracy: 0.4567 - final_authenticity_accuracy: 0.7890 - val_loss: 1.1234 - val_ensemble_classification_loss: 0.8765 - val_final_authenticity_loss: 0.2469 - val_ensemble_classification_accuracy: 0.5000 - val_final_authenticity_accuracy: 0.8000
```

### Step 4: Training Outputs

After training completes, you'll find:
- **Model weights**: `models/kyc_validator.h5` (or your custom path)
- **Confusion matrix**: `confusion_matrix.png`
- **Training history**: `training_history.png`

### Step 5: Monitor Training

**Option A: Use TensorBoard (if configured)**
```bash
tensorboard --logdir=logs
```

**Option B: Check Training History Plot**
After training, view `training_history.png` to see:
- Loss curves (should decrease)
- Accuracy curves (should increase)
- Overfitting indicators

### Step 6: Training Tips

1. **Start Small**: Begin with 5-10 epochs to test
2. **Monitor Validation**: Watch for overfitting
3. **Adjust Batch Size**: 
   - Larger batch (32-64): Faster, more memory
   - Smaller batch (8-16): Slower, less memory
4. **Learning Rate**: 
   - Default 0.001 works well
   - Lower (0.0001) for fine-tuning
   - Higher (0.01) may cause instability

---

## Testing the Model

### Method 1: Quick Test with Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Open Notebook**:
   - Navigate to `notebooks/quick_test.ipynb`
   - Run cells sequentially

3. **Test Components**:
   - Model creation
   - OCR extraction
   - Fake detection
   - Full pipeline

### Method 2: Test with Python Script

Create a test script `test_model.py`:

```python
import sys
import os
sys.path.append('src')

import numpy as np
import cv2
from models import create_ensemble_model, compile_model
from ocr_utils import extract_document_info
from fake_detector import comprehensive_fake_detection

# Load model
model = create_ensemble_model(input_shape=(150, 150, 3), num_classes=4)
model.load_weights('models/kyc_validator.h5')

# Load test image
image_path = 'data/test/aadhaar/sample.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess
preprocessed = cv2.resize(image_rgb, (150, 150))
preprocessed = preprocessed.astype(np.float32) / 255.0
input_batch = np.expand_dims(preprocessed, axis=0)

# Predict
predictions = model.predict(input_batch, verbose=0)
class_probs = predictions[0][0]
auth_score = predictions[1][0][0]

class_names = ['Aadhaar', 'PAN', 'Fake', 'Other']
predicted_class = class_names[np.argmax(class_probs)]
confidence = class_probs[np.argmax(class_probs)]

print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
print(f"Authenticity: {auth_score:.4f}")

# OCR
ocr_info = extract_document_info(image)
print(f"Aadhaar: {ocr_info['aadhaar_number']}")
print(f"PAN: {ocr_info['pan_number']}")

# Fake detection
fake_result = comprehensive_fake_detection(image, predicted_class.lower(), ocr_info['raw_text'])
print(f"Is Fake: {fake_result['is_fake']}")
print(f"Issues: {fake_result['issues']}")
```

Run it:
```bash
python test_model.py
```

### Method 3: Evaluate on Test Set

The training script automatically evaluates on test set if available. Check:
- Confusion matrix
- Classification report
- Test accuracy metrics

---

## Using the Streamlit App

### Step 1: Start Streamlit Server

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 2: Load the Model

1. In the sidebar, enter model path: `models/kyc_validator.h5`
2. Click "Load Model"
3. Wait for "Model loaded successfully!" message

### Step 3: Upload Document

1. Click "Browse files" or drag and drop
2. Supported formats: PNG, JPG, JPEG, PDF
3. Wait for processing

### Step 4: View Results

The app displays:
- **Original document** (left)
- **Analysis results** with highlights (right)
- **Classification metrics**: Type, confidence, authenticity
- **Extracted information**: Aadhaar/PAN numbers
- **Fake detection**: Issues and authenticity score
- **JSON output**: Downloadable results

### Step 5: Interpret Results

**Classification Results:**
- **Document Type**: Aadhaar, PAN, Fake, or Other
- **Type Confidence**: How confident the model is (0-100%)
- **Authenticity Score**: Likelihood document is real (0-100%)
- **Status**: ✅ Real or ❌ Fake

**Fake Detection Issues:**
- `color_mismatch`: Document doesn't have expected colors
- `plain_paper_detected`: Looks like plain white paper
- `tampered_borders`: Suspicious edges detected
- `handwritten_on_plain_paper`: Handwritten text on plain background
- `no_qr_code`: Missing QR code
- `invalid_qr_format`: QR code format incorrect
- `suspicious_lines`: Too many straight lines
- `incomplete_document`: Document structure incomplete
- `irregular_shape`: Document shape is unusual

---

## Complete Workflow Example

### Day 1: Setup and Data Collection
```bash
# 1. Setup environment
cd KYC-Doc-Validator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Verify installation
python -c "import tensorflow; print('OK')"

# 3. Prepare dataset structure
mkdir -p data/{train,val,test}/{aadhaar,pan,fake,other}

# 4. Add your images to respective folders
# (Copy images manually or use scripts)
```

### Day 2: Training
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Start training
python src/train.py --data_dir data --epochs 10 --batch_size 32

# 3. Monitor training progress
# 4. Check outputs: confusion_matrix.png, training_history.png
```

### Day 3: Testing
```bash
# 1. Test with notebook
jupyter notebook notebooks/quick_test.ipynb

# 2. Test with Streamlit
streamlit run app/streamlit_app.py

# 3. Upload test documents and verify results
```

---

## Troubleshooting

### Issue: Tesseract Not Found
**Error**: `TesseractNotFoundError`

**Solution**:
```bash
# macOS
brew install tesseract

# Linux
sudo apt-get install tesseract-ocr

# Verify
which tesseract
tesseract --version
```

### Issue: Out of Memory During Training
**Error**: `ResourceExhaustedError` or system freezes

**Solution**:
```bash
# Reduce batch size
python src/train.py --batch_size 8

# Or reduce image size in models.py (change 150x150 to 128x128)
```

### Issue: No Training Data Found
**Error**: `No training data found!`

**Solution**:
1. Check dataset structure matches expected format
2. Verify images are in correct folders
3. Check image file extensions (.jpg, .jpeg, .png)
4. Run: `find data/train -type f | head -5` to verify files exist

### Issue: Model File Not Found in Streamlit
**Error**: `Model file not found`

**Solution**:
1. Train the model first: `python src/train.py`
2. Check model exists: `ls models/`
3. Update model path in Streamlit sidebar

### Issue: Poor OCR Results
**Solution**:
1. Ensure images are clear and high resolution
2. Preprocess images (adjust contrast, denoise)
3. Check Tesseract language packs: `tesseract --list-langs`
4. Install additional language data if needed

### Issue: Low Training Accuracy
**Solution**:
1. Increase dataset size
2. Add more data augmentation
3. Train for more epochs
4. Adjust learning rate
5. Check for class imbalance

### Issue: Import Errors
**Error**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade

# Check virtual environment is activated
which python  # Should show venv path
```

---

## Next Steps After Training

1. **Evaluate Performance**:
   - Review confusion matrix
   - Check per-class accuracy
   - Identify weak classes

2. **Improve Model**:
   - Collect more data for weak classes
   - Fine-tune hyperparameters
   - Try different architectures

3. **Deploy**:
   - Set up production environment
   - Replace mock APIs with real ones
   - Add logging and monitoring

4. **Optimize**:
   - Model quantization for faster inference
   - Batch processing for multiple documents
   - Caching for repeated queries

---

## Quick Reference Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Training
python src/train.py --data_dir data --epochs 10

# Testing
streamlit run app/streamlit_app.py

# Notebook
jupyter notebook notebooks/quick_test.ipynb

# Check dataset
find data/train -type f | wc -l

# Verify model
ls -lh models/
```

---

## Support

For issues or questions:
1. Check this guide
2. Review README.md
3. Check code comments
4. Review error messages carefully

Happy Training! 🚀

