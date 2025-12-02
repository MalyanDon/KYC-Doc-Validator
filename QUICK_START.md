# Quick Start Guide - KYC Document Validator

This is a condensed guide to get you up and running quickly. For detailed instructions, see `TRAINING_GUIDE.md`.

## 🚀 5-Minute Quick Start

### Step 1: Setup (2 minutes)

```bash
# Navigate to project
cd "/Users/abhishekmalyan/Aadhar Card and Pancard/KYC-Doc-Validator"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import tensorflow; print('✅ TensorFlow OK')"
python -c "import cv2; print('✅ OpenCV OK')"
tesseract --version  # Should show version
```

### Step 2: Prepare Dataset (2 minutes)

```bash
# Create dataset structure
python prepare_dataset.py --create

# Add your images to:
# - data/train/aadhaar/  (real Aadhaar cards)
# - data/train/pan/      (real PAN cards)
# - data/train/fake/     (fake documents)
# - data/train/other/    (other documents)

# Repeat for data/val/ and data/test/

# Verify dataset
python prepare_dataset.py --count --verify
```

**Minimum dataset:**
- At least 10-20 images per class per split
- Total: ~120-240 images minimum
- Recommended: 200+ images per class

### Step 3: Train Model (5-30 minutes)

```bash
# Basic training (10 epochs)
python src/train.py --data_dir data --epochs 10

# With custom settings
python src/train.py --data_dir data --epochs 20 --batch_size 16 --learning_rate 0.0001
```

**What happens:**
- Model loads and compiles
- Training starts (shows progress bars)
- Best model saved to `models/kyc_validator.h5`
- Confusion matrix and training plots generated

### Step 4: Test Model (1 minute)

**Option A: Command Line**
```bash
# Test single image
python test_model.py --image path/to/test_image.jpg

# Test all images in directory
python test_model.py --dir data/test/aadhaar/
```

**Option B: Streamlit App**
```bash
streamlit run app/streamlit_app.py
```
Then:
1. Load model in sidebar
2. Upload image/PDF
3. View results

**Option C: Jupyter Notebook**
```bash
jupyter notebook notebooks/quick_test.ipynb
```

---

## 📋 Complete Workflow Checklist

### ✅ Pre-Training Checklist
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Tesseract OCR installed and verified
- [ ] Dataset structure created (`python prepare_dataset.py --create`)
- [ ] Images added to data folders
- [ ] Dataset verified (`python prepare_dataset.py --verify`)

### ✅ Training Checklist
- [ ] Dataset has images in all required folders
- [ ] Training command ready
- [ ] Sufficient disk space (models can be 100-500MB)
- [ ] Training started
- [ ] Monitoring training progress
- [ ] Model saved successfully

### ✅ Post-Training Checklist
- [ ] Model file exists (`ls models/kyc_validator.h5`)
- [ ] Confusion matrix generated
- [ ] Training history plot generated
- [ ] Tested on sample images
- [ ] Streamlit app working

---

## 🎯 Common Use Cases

### Use Case 1: Quick Test with Existing Model
```bash
# If you have a trained model
streamlit run app/streamlit_app.py
# Upload image and test
```

### Use Case 2: Train from Scratch
```bash
# 1. Prepare data
python prepare_dataset.py --create
# (Add images manually)

# 2. Train
python src/train.py --data_dir data --epochs 10

# 3. Test
python test_model.py --image data/test/aadhaar/sample.jpg
```

### Use Case 3: Test Multiple Images
```bash
# Test all images in a folder
python test_model.py --dir data/test/aadhaar/
```

### Use Case 4: Experiment with Hyperparameters
```bash
# Try different learning rates
python src/train.py --epochs 10 --learning_rate 0.0001
python src/train.py --epochs 10 --learning_rate 0.001
python src/train.py --epochs 10 --learning_rate 0.01

# Try different batch sizes
python src/train.py --epochs 10 --batch_size 8   # Smaller, slower
python src/train.py --epochs 10 --batch_size 32  # Default
python src/train.py --epochs 10 --batch_size 64  # Larger, faster (needs more RAM)
```

---

## 🔧 Troubleshooting Quick Fixes

### Problem: "TesseractNotFoundError"
```bash
# macOS
brew install tesseract

# Linux
sudo apt-get install tesseract-ocr

# Verify
tesseract --version
```

### Problem: "No training data found!"
```bash
# Check dataset
python prepare_dataset.py --count

# Verify structure
ls -R data/train/
```

### Problem: Out of Memory
```bash
# Reduce batch size
python src/train.py --batch_size 8
```

### Problem: Model not found in Streamlit
```bash
# Train first
python src/train.py --data_dir data --epochs 10

# Check model exists
ls models/
```

---

## 📊 Understanding Results

### Training Output
- **Loss**: Should decrease over time
- **Accuracy**: Should increase over time
- **Validation metrics**: Should track training metrics

### Prediction Results
- **Type**: Aadhaar, PAN, Fake, or Other
- **Confidence**: 0-100% (higher is better)
- **Authenticity**: 0-100% (higher = more likely real)
- **Issues**: List of detected problems

### Good Results
- ✅ Classification accuracy > 80%
- ✅ Authenticity score > 0.7 for real documents
- ✅ Authenticity score < 0.5 for fake documents
- ✅ Few false positives/negatives

---

## 🎓 Next Steps

1. **Improve Dataset**: Add more diverse images
2. **Fine-tune**: Adjust hyperparameters
3. **Evaluate**: Test on real-world documents
4. **Deploy**: Set up production environment
5. **Monitor**: Track performance over time

---

## 📚 Additional Resources

- **Detailed Guide**: See `TRAINING_GUIDE.md`
- **Project README**: See `README.md`
- **Code Documentation**: Check docstrings in source files
- **Notebook**: Explore `notebooks/quick_test.ipynb`

---

## 💡 Pro Tips

1. **Start Small**: Begin with 5 epochs to test pipeline
2. **Monitor Validation**: Watch for overfitting
3. **Save Checkpoints**: Models are auto-saved during training
4. **Use GPU**: If available, TensorFlow will use it automatically
5. **Data Augmentation**: Already included in training script
6. **Test Regularly**: Test after each training session

---

**Ready to start?** Run the commands in Step 1! 🚀

