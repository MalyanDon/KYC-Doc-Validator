# 🚀 Training Status

## Current Training Run

**Started:** Fresh training from scratch  
**Dataset:**
- **Train:** 1,575 Aadhaar + 1,458 PAN = **3,033 images**
- **Validation:** 277 Aadhaar + 268 PAN = **545 images**
- **Test:** 265 Aadhaar images

**Model Configuration:**
- **Classes:** Aadhaar, PAN (Fake/Other will be added when data is available)
- **Architecture:** Ensemble CNN (VGG16 + Custom CNN + Sequential CNN)
- **Input Size:** 150x150x3
- **Batch Size:** 32
- **Epochs:** 10 (with early stopping)

## Monitor Training

### Real-time Progress
```bash
tail -f training_fresh.log
```

### Check Training Status
```bash
# Check if training is running
pgrep -f "train.py" && echo "✅ Training is running" || echo "❌ Training stopped"

# View latest progress
tail -30 training_fresh.log | grep -E "(Epoch|loss|accuracy|val_)"
```

### Expected Training Time
- **Per Epoch:** ~60-90 seconds (depending on CPU)
- **Total (10 epochs):** ~10-15 minutes
- **With Early Stopping:** May stop earlier if validation loss doesn't improve

## Training Outputs

Once training completes, you'll have:
1. **Model:** `models/kyc_validator.h5`
2. **Confusion Matrix:** `confusion_matrix.png`
3. **Training History Plot:** `training_history.png`
4. **Training Log:** `training_fresh.log`

## Next Steps After Training

1. **Evaluate the model:**
   ```bash
   python src/test_model.py --model models/kyc_validator.h5 --image path/to/test/image.jpg
   ```

2. **Add Fake/Other data** (when available):
   - Add images to `data/train/fake/` and `data/train/other/`
   - Re-train the model to include all 4 classes

3. **Use in Streamlit app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Troubleshooting

If training stops unexpectedly:
1. Check `training_fresh.log` for errors
2. Verify dataset structure: `ls -R data/train/`
3. Check available disk space: `df -h`
4. Restart training: `python src/train.py --data_dir data --epochs 10 --batch_size 32`

