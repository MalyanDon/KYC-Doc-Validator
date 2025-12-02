# 🎯 Training Objectives - What to Train Your Model For

## Overview

Your KYC Document Validator has **multiple training components**. Here's what each one does and when to train them:

---

## 📋 **Training Components**

### **1. Main Ensemble CNN Model** ⭐ **PRIMARY TRAINING**

**What it does:**
- **Classification:** Identifies document type (Aadhaar, PAN, Fake, Other)
- **Authenticity:** Detects if document is real or fake

**What it learns:**
- Visual patterns (colors, textures, shapes)
- Document layout differences (implicitly)
- How to distinguish between document types
- How to detect fake documents

**Training Command:**
```powershell
.\venv\Scripts\Activate.ps1
python src/train.py --data_dir data --epochs 10 --batch_size 32
```

**Required Data:**
- ✅ Aadhaar images (you have: 1,575 train, 277 val, 265 test)
- ✅ PAN images (you have: 1,283 train, 193 val, 250 test)
- ⚠️ Fake images (you have: 0 - needed for fake detection)
- ⚠️ Other images (you have: 0 - needed for 4-class classification)

**Output:**
- `models/kyc_validator.h5` - Trained model weights
- `confusion_matrix.png` - Classification performance
- `training_history.png` - Training curves

**Status:** ✅ **READY TO TRAIN** (with Aadhaar/PAN data)

---

### **2. Position Detector Training** 🔧 **OPTIONAL BUT RECOMMENDED**

**What it does:**
- Learns where elements are positioned in documents
- Photo location (normalized coordinates)
- Text field positions (name, DOB, document number)
- Used by layout validator to check document structure

**What it learns:**
- Exact photo position (x, y, width, height)
- Text field positions
- Layout patterns from your actual documents

**Training Command:**
```powershell
.\venv\Scripts\Activate.ps1

# Train Aadhaar positions
python train_positions.py --method images --input_dir data/train/aadhaar/ --doc_type aadhaar

# Train PAN positions
python train_positions.py --method images --input_dir data/train/pan/ --doc_type pan
```

**Required Data:**
- ✅ Aadhaar images (you have: 1,575 images)
- ✅ PAN images (you have: 1,283 images)

**Output:**
- `models/learned_aadhaar_positions.json` - Learned Aadhaar positions
- `models/learned_pan_positions.json` - Learned PAN positions

**Status:** ✅ **READY TO TRAIN** (improves layout validation accuracy)

---

### **3. Enhanced Model Training** 🚀 **ADVANCED (Optional)**

**What it does:**
- **Multi-task learning:** Classification + Authenticity + Position Prediction
- Predicts document type, authenticity, AND element positions simultaneously
- More integrated approach

**What it learns:**
- Everything from Main Model (classification + authenticity)
- PLUS: Element positions (photo, text fields)
- End-to-end learning of structure

**Training Command:**
```powershell
.\venv\Scripts\Activate.ps1
python src/train_enhanced.py --data_dir data --epochs 10 --batch_size 32
```

**Required Data:**
- Same as Main Model (Aadhaar, PAN, Fake, Other)

**Output:**
- `models/kyc_validator_enhanced.h5` - Enhanced model with position prediction

**Status:** ⚠️ **ADVANCED** (use after main model is trained)

---

## 🎯 **Recommended Training Order**

### **Step 1: Train Position Detector** (5-10 minutes)
**Why first?** Improves layout validation accuracy

```powershell
.\venv\Scripts\Activate.ps1

# Train Aadhaar positions
python train_positions.py --method images --input_dir data/train/aadhaar/ --doc_type aadhaar

# Train PAN positions  
python train_positions.py --method images --input_dir data/train/pan/ --doc_type pan
```

**Result:** Creates `learned_aadhaar_positions.json` and `learned_pan_positions.json`

---

### **Step 2: Train Main Ensemble Model** (30-60 minutes)
**Why second?** Core functionality - classification and fake detection

```powershell
.\venv\Scripts\Activate.ps1
python src/train.py --data_dir data --epochs 10 --batch_size 32
```

**What happens:**
- Model learns to classify Aadhaar vs PAN
- Model learns authenticity detection (currently limited without fake data)
- Saves trained model to `models/kyc_validator.h5`

**Note:** Will work with 2 classes (Aadhaar/PAN) even though model expects 4 classes

---

### **Step 3: Test Your Model** (5 minutes)
**Verify everything works:**

```powershell
# Test with Streamlit app
streamlit run app/streamlit_app.py

# OR test from command line
python test_model.py --image data/test/aadhaar/sample.jpg
```

---

### **Step 4: Enhanced Training (Optional)** (30-60 minutes)
**Only if you want position prediction integrated:**

```powershell
python src/train_enhanced.py --data_dir data --epochs 10
```

---

## 📊 **What Each Model Can Do**

### **Main Model (After Training):**
- ✅ Classify documents: Aadhaar vs PAN
- ✅ Detect authenticity: Real vs Fake (limited without fake data)
- ✅ Extract text using OCR
- ✅ Validate document numbers (Aadhaar/PAN format)
- ⚠️ Layout validation (uses learned positions if trained)

### **Position Detector (After Training):**
- ✅ Knows exact photo positions
- ✅ Knows text field positions
- ✅ Validates document structure
- ✅ Detects layout tampering

### **Enhanced Model (After Training):**
- ✅ Everything Main Model does
- ✅ PLUS: Predicts element positions directly
- ✅ More integrated approach

---

## 🎯 **Current Training Status**

| Component | Status | Data Available | Can Train? |
|-----------|--------|----------------|------------|
| **Main Model** | ⏳ Not Trained | ✅ Aadhaar + PAN | ✅ **YES** |
| **Position Detector** | ⏳ Not Trained | ✅ Aadhaar + PAN | ✅ **YES** |
| **Enhanced Model** | ⏳ Not Trained | ✅ Aadhaar + PAN | ✅ **YES** |

---

## 💡 **Quick Start Training**

**Minimum Training (Get Started Fast):**
```powershell
.\venv\Scripts\Activate.ps1

# 1. Train positions (5 min)
python train_positions.py --method images --input_dir data/train/aadhaar/ --doc_type aadhaar
python train_positions.py --method images --input_dir data/train/pan/ --doc_type pan

# 2. Train main model (30-60 min)
python src/train.py --data_dir data --epochs 10 --batch_size 32

# 3. Test it
streamlit run app/streamlit_app.py
```

**Full Training (Best Results):**
1. Add fake document images to `data/train/fake/`, `data/val/fake/`, `data/test/fake/`
2. Add other document images to `data/train/other/`, `data/val/other/`, `data/test/other/`
3. Train position detector
4. Train main model
5. Test and evaluate

---

## 📝 **Summary**

**You should train:**

1. **Position Detector** ✅ **RECOMMENDED FIRST**
   - Improves layout validation
   - Uses your actual document data
   - Quick (5-10 minutes)

2. **Main Ensemble Model** ✅ **REQUIRED**
   - Core functionality
   - Classification + Fake Detection
   - Takes 30-60 minutes

3. **Enhanced Model** ⚠️ **OPTIONAL**
   - Advanced multi-task learning
   - Only if you want integrated position prediction

**Current Status:** Ready to train with your Aadhaar/PAN dataset! 🚀

