# Training Position Detector - Guide

## 🎯 Why Train Instead of Guessing?

You're absolutely right! Instead of guessing positions, we should **learn them from real documents**.

---

## 📚 Two Approaches

### Approach 1: Learn from Annotated Data (Best)

**Step 1: Annotate Real Documents**
```bash
# Use annotation tool
python src/annotation_helper.py image1.jpg
# Or batch annotate
python src/annotation_helper.py --batch data/train/aadhaar/ annotations/
```

**Step 2: Learn Positions**
```python
from src.trainable_layout_detector import LayoutPositionLearner

learner = LayoutPositionLearner(doc_type='aadhaar')
learned = learner.learn_from_annotations('annotations/')
learner.save_learned_positions('models/learned_aadhaar_positions.json')
```

**Step 3: Use Learned Positions**
```python
# Layout validator automatically uses learned positions if available
from src.layout_validator import comprehensive_layout_validation

result = comprehensive_layout_validation(image, 'aadhaar', ocr_text)
# Uses learned positions automatically!
```

### Approach 2: Auto-Learn from Images (Quick Start)

**Automatically detect positions from real documents:**
```python
from src.trainable_layout_detector import LayoutPositionLearner

learner = LayoutPositionLearner(doc_type='aadhaar')
learned = learner.learn_from_images('data/train/aadhaar/', auto_annotate=True)
learner.save_learned_positions('models/learned_aadhaar_positions.json')
```

**How it works:**
- Uses face detection to find photos
- Uses OCR to find text regions
- Calculates average positions
- Learns from multiple samples

---

## 🛠️ Annotation Tool Usage

### Interactive Annotation

```bash
python src/annotation_helper.py document.jpg
```

**Controls:**
- `p` - Annotate PHOTO
- `n` - Annotate NAME
- `d` - Annotate DOB
- `a` - Annotate AADHAAR NUMBER
- Click and drag to draw box
- `s` - Save annotation
- `q` - Quit

**Output:** Creates `document.json` with bounding boxes

### Batch Annotation

```bash
python src/annotation_helper.py --batch data/train/aadhaar/ annotations/
```

Annotates all images in directory.

---

## 📊 Annotation Format

**JSON file created:**
```json
{
  "image_path": "document.jpg",
  "image_size": [600, 800],
  "photo": {
    "x": 50,
    "y": 120,
    "width": 150,
    "height": 180
  },
  "text_regions": {
    "name": {
      "x": 220,
      "y": 165,
      "width": 200,
      "height": 30
    },
    "aadhaar_number": {
      "x": 220,
      "y": 400,
      "width": 300,
      "height": 30
    }
  }
}
```

---

## 🎓 Learning Process

### Step 1: Collect Real Documents
- Gather 20-50 real Aadhaar/PAN cards
- Place in `data/train/aadhaar/` or `data/train/pan/`

### Step 2: Annotate (Optional but Recommended)
```bash
# Annotate a few samples manually
python src/annotation_helper.py data/train/aadhaar/sample1.jpg
python src/annotation_helper.py data/train/aadhaar/sample2.jpg
# ... repeat for 5-10 samples
```

### Step 3: Learn Positions
```python
from src.trainable_layout_detector import LayoutPositionLearner

# Method 1: From annotations (more accurate)
learner = LayoutPositionLearner(doc_type='aadhaar')
if os.path.exists('annotations/'):
    learned = learner.learn_from_annotations('annotations/')
else:
    # Method 2: Auto-learn from images
    learned = learner.learn_from_images('data/train/aadhaar/')

learner.save_learned_positions('models/learned_aadhaar_positions.json')
```

### Step 4: Verify Learned Positions
```python
positions = learner.get_positions()
print(f"Photo position: {positions['photo_region']}")
print(f"Name position: {positions['text_regions']['name']}")
```

---

## 📈 Statistics

The learner also tracks statistics:
```python
stats = learner.position_stats
print(f"Photo: Mean={stats['photo']['mean']}, Std={stats['photo']['std']}")
print(f"Learned from {stats['photo']['count']} samples")
```

**Useful for:**
- Understanding variation in real documents
- Setting appropriate thresholds
- Detecting outliers

---

## 🔄 Automatic Usage

Once positions are learned and saved, the layout validator **automatically uses them**:

```python
from src.layout_validator import comprehensive_layout_validation

# Automatically loads learned positions if available
result = comprehensive_layout_validation(image, 'aadhaar', ocr_text)
```

**Priority:**
1. ✅ Learned positions (if `models/learned_aadhaar_positions.json` exists)
2. ⚠️ Default positions (fallback)

---

## 🎯 Best Practices

1. **Annotate 10-20 samples** for better accuracy
2. **Use real documents** from your use case
3. **Include variations** (different formats, ages, etc.)
4. **Verify learned positions** before using
5. **Update periodically** as you get more data

---

## 💡 Example Workflow

```python
# 1. Learn positions from your real documents
from src.trainable_layout_detector import LayoutPositionLearner

learner = LayoutPositionLearner(doc_type='aadhaar')
learned = learner.learn_from_images('data/train/aadhaar/')
learner.save_learned_positions('models/learned_aadhaar_positions.json')

# 2. Use in validation (automatic)
from src.layout_validator import comprehensive_layout_validation
result = comprehensive_layout_validation(image, 'aadhaar', ocr_text)

# 3. Check if using learned positions
print(f"Using learned positions: {result.get('using_learned', False)}")
```

---

## 🚀 Quick Start

```bash
# 1. Learn from your real documents
python -c "
from src.trainable_layout_detector import LayoutPositionLearner
learner = LayoutPositionLearner('aadhaar')
learner.learn_from_images('data/train/aadhaar/')
learner.save_learned_positions('models/learned_aadhaar_positions.json')
"

# 2. Verify
python -c "
from src.trainable_layout_detector import LayoutPositionLearner
learner = LayoutPositionLearner('aadhaar')
learner.load_learned_positions('models/learned_aadhaar_positions.json')
print(learner.get_positions())
"
```

---

**Now positions are learned from real data, not guessed!** 🎉

