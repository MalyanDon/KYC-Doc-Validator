# Start Training - Step-by-Step Guide

## 🎯 Complete Training Process

Follow these steps in order to train your KYC Document Validator.

---

## Step 1: Prepare Your Dataset (15-30 minutes)

### 1.1 Create Dataset Structure
```bash
cd "/Users/abhishekmalyan/Aadhar Card and Pancard/KYC-Doc-Validator"
python prepare_dataset.py --create
```

### 1.2 Add Your Images

**Organize images into folders:**

```
data/
├── train/
│   ├── aadhaar/    ← Add real Aadhaar card images here
│   ├── pan/        ← Add real PAN card images here
│   ├── fake/       ← Add fake/synthetic documents here
│   └── other/      ← Add other document types here
├── val/
│   └── (same structure - validation images)
└── test/
    └── (same structure - test images)
```

**Minimum Requirements:**
- At least 10-20 images per class per split
- Total: ~120-240 images minimum
- Recommended: 50+ images per class

**How to add images:**
```bash
# Copy your images to the folders
cp /path/to/aadhaar_images/*.jpg data/train/aadhaar/
cp /path/to/pan_images/*.jpg data/train/pan/
cp /path/to/fake_images/*.jpg data/train/fake/
```

### 1.3 Verify Dataset
```bash
python prepare_dataset.py --count --verify
```

**Expected output:**
```
📊 Dataset Statistics:
TRAIN:
  aadhaar:   50 images
  pan:       50 images
  fake:      30 images
  other:     20 images
  TOTAL:     150 images
...
```

---

## Step 2: Train Position Detector (Optional but Recommended - 10 minutes)

### 2.1 Quick Auto-Learn (Recommended for Start)

```bash
# Learn Aadhaar positions
python train_positions.py \
    --method images \
    --input_dir data/train/aadhaar/ \
    --doc_type aadhaar

# Learn PAN positions
python train_positions.py \
    --method images \
    --input_dir data/train/pan/ \
    --doc_type pan
```

**What this does:**
- Automatically detects photos and text in your real documents
- Learns average positions
- Saves to `models/learned_aadhaar_positions.json`

**Time:** ~5-10 minutes depending on number of images

### 2.2 Verify Learned Positions (Optional)

```python
python -c "
from src.trainable_layout_detector import LayoutPositionLearner
learner = LayoutPositionLearner('aadhaar')
learner.load_learned_positions('models/learned_aadhaar_positions.json')
print('Learned positions:', learner.get_positions())
"
```

---

## Step 3: Train the Main Classification Model (30-60 minutes)

### 3.1 Basic Training

```bash
# Activate virtual environment
source venv/bin/activate

# Start training
python src/train.py --data_dir data --epochs 10
```

**What happens:**
- Loads images from `data/train/`, `data/val/`, `data/test/`
- Creates ensemble model (VGG16 + Custom CNN + Sequential)
- Trains for 10 epochs
- Saves best model to `models/kyc_validator.h5`
- Generates confusion matrix and training plots

**Expected output:**
```
Loading dataset...
Training samples: 150
Validation samples: 30
Test samples: 20

Creating ensemble model...
Starting training...

Epoch 1/10
5/5 [==============================] - 45s 9s/step - loss: 1.2345 ...
...
```

### 3.2 Advanced Training (Optional)

```bash
# Custom parameters
python src/train.py \
    --data_dir data \
    --epochs 20 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --model_save_path models/kyc_validator_v1.h5
```

**Parameters:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32, reduce if out of memory)
- `--learning_rate`: Learning rate (default: 0.001)
- `--model_save_path`: Where to save model

---

## Step 4: Verify Training Results (5 minutes)

### 4.1 Check Outputs

After training, you should have:

```bash
# Check model file
ls -lh models/kyc_validator.h5

# Check plots
ls -lh confusion_matrix.png training_history.png
```

### 4.2 Review Training History

Open `training_history.png` to see:
- Loss curves (should decrease)
- Accuracy curves (should increase)
- Validation metrics

### 4.3 Review Confusion Matrix

Open `confusion_matrix.png` to see:
- Classification performance per class
- Identify weak classes

---

## Step 5: Test Your Model (5 minutes)

### 5.1 Test with Command Line

```bash
# Test single image
python test_model.py --image data/test/aadhaar/sample.jpg

# Test all images in folder
python test_model.py --dir data/test/aadhaar/
```

### 5.2 Test with Streamlit

```bash
streamlit run app/streamlit_app.py
```

Then:
1. Load model: Enter `models/kyc_validator.h5` in sidebar
2. Upload test image
3. View results

---

## 📋 Complete Checklist

### Pre-Training:
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset structure created (`python prepare_dataset.py --create`)
- [ ] Images added to data folders
- [ ] Dataset verified (`python prepare_dataset.py --verify`)

### Training:
- [ ] Position detector trained (optional: `python train_positions.py`)
- [ ] Main model trained (`python src/train.py`)
- [ ] Model file exists (`models/kyc_validator.h5`)
- [ ] Training plots generated

### Post-Training:
- [ ] Model tested on sample images
- [ ] Streamlit app working
- [ ] Results reviewed

---

## 🚀 Quick Start Commands

**Copy and paste these commands:**

```bash
# 1. Navigate to project
cd "/Users/abhishekmalyan/Aadhar Card and Pancard/KYC-Doc-Validator"

# 2. Activate environment
source venv/bin/activate

# 3. Create dataset structure
python prepare_dataset.py --create

# 4. Add your images to data/train/, data/val/, data/test/
# (Do this manually - copy images to folders)

# 5. Verify dataset
python prepare_dataset.py --count --verify

# 6. Train position detector (optional)
python train_positions.py --method images --input_dir data/train/aadhaar/ --doc_type aadhaar

# 7. Train main model
python src/train.py --data_dir data --epochs 10

# 8. Test model
python test_model.py --image data/test/aadhaar/sample.jpg

# 9. Run Streamlit app
streamlit run app/streamlit_app.py
```

---

## ⏱️ Time Estimates

| Step | Time | Notes |
|------|------|-------|
| Dataset Prep | 15-30 min | Depends on number of images |
| Position Training | 5-10 min | Optional, auto-learn |
| Main Training | 30-60 min | Depends on dataset size, epochs |
| Testing | 5-10 min | Quick verification |
| **Total** | **~1-2 hours** | First time setup |

---

## 🎯 What You Need

### Minimum:
- ✅ 10-20 images per class (Aadhaar, PAN, Fake, Other)
- ✅ Split into train/val/test (70/15/15 ratio)
- ✅ Total: ~120-240 images

### Recommended:
- ✅ 50+ images per class
- ✅ More diverse samples
- ✅ Total: ~600+ images

### Where to Get Images:
1. **Real documents** (with permission)
2. **Public datasets** (if available)
3. **Synthetic fakes** (create using augmentation)
4. **Cloned repositories** (temp_magnum, temp_doc - if they have data)

---

## 🆘 Troubleshooting

### Problem: "No training data found!"
**Solution:**
```bash
# Check dataset
python prepare_dataset.py --count

# Verify images are in correct folders
ls data/train/aadhaar/
ls data/train/pan/
```

### Problem: Out of Memory
**Solution:**
```bash
# Reduce batch size
python src/train.py --data_dir data --epochs 10 --batch_size 8
```

### Problem: Training too slow
**Solution:**
- Reduce number of epochs for testing
- Use smaller dataset initially
- Check if GPU is available (TensorFlow will use it automatically)

---

## 📚 Next Steps After Training

1. **Evaluate Results**
   - Review confusion matrix
   - Check per-class accuracy
   - Identify weak classes

2. **Improve Model**
   - Add more data for weak classes
   - Adjust hyperparameters
   - Train for more epochs

3. **Deploy**
   - Use Streamlit app
   - Integrate into your system
   - Replace mock APIs with real ones

---

**Ready to start? Begin with Step 1!** 🚀

