# KYC Document Validator

A comprehensive Python project for classifying and validating Indian ID documents (Aadhaar vs PAN) with fake detection capabilities. The system uses an ensemble CNN approach combining VGG16, Custom CNN, and Lightweight Sequential models for robust document classification and authenticity verification.

## Features

- **Multi-Model Ensemble**: Combines 3 CNN backbones (VGG16, Custom CNN, Sequential) for robust classification
- **Document Classification**: Classifies documents as Aadhaar, PAN, Fake, or Other
- **OCR Text Extraction**: Extracts text using Tesseract OCR and validates document numbers
- **Fake Detection**: Multiple detection methods including:
  - Color histogram analysis (Aadhaar blue tint detection)
  - Edge detection for tampered borders
  - Handwritten number detection on plain paper
  - QR code validation
  - Layout tampering detection
- **Streamlit UI**: User-friendly web interface for document upload and validation
- **PDF Support**: Handles both image and PDF document uploads
- **JSON Output**: Structured results with confidence scores and detected issues

## Project Structure

```
KYC-Doc-Validator/
├── src/
│   ├── models.py          # Ensemble CNN model definitions
│   ├── ocr_utils.py       # OCR and text extraction utilities
│   ├── fake_detector.py   # Fake detection algorithms
│   └── train.py           # Training script
├── app/
│   └── streamlit_app.py   # Streamlit web application
├── data/
│   ├── train/
│   │   ├── aadhaar/
│   │   ├── pan/
│   │   ├── fake/
│   │   └── other/
│   ├── val/
│   │   └── (same structure)
│   └── test/
│       └── (same structure)
├── notebooks/
│   └── quick_test.ipynb   # Sample notebook for quick testing
├── models/                 # Trained model weights (created after training)
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system
  - **macOS**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`
  - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Setup

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd KYC-Doc-Validator
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**:
   - Organize images in the `data/` directory following the structure above
   - Place Aadhaar, PAN, fake, and other document images in respective folders
   - Recommended: ~1K images total across train/val/test splits

## Usage

### Training the Model

Train the ensemble model on your dataset:

```bash
python src/train.py --data_dir data --epochs 10 --batch_size 32 --learning_rate 0.001
```

Arguments:
- `--data_dir`: Path to data directory (default: `data`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--model_save_path`: Path to save trained model (default: `models/kyc_validator.h5`)

The training script will:
- Load and preprocess images from the dataset
- Train the ensemble model with data augmentation
- Evaluate on test set and generate confusion matrix
- Save the best model weights

### Running the Streamlit App

1. **Start the Streamlit application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **In the web interface**:
   - Load the trained model (use sidebar)
   - Upload an image or PDF document
   - View classification results, OCR extraction, and fake detection analysis
   - Download JSON results

### Quick Test with Notebook

Open `notebooks/quick_test.ipynb` for a quick test of the model and utilities.

## Model Architecture

### Ensemble Model

The ensemble combines three CNN backbones:

1. **VGG16 Backbone**: Pre-trained VGG16 with custom classification and authenticity heads
2. **Custom CNN Backbone**: 5-layer CNN with batch normalization and dropout
3. **Sequential CNN Backbone**: Lightweight 5-layer sequential CNN (32→64→128→256→512 filters)

The ensemble averages outputs from all three models for final predictions.

### Outputs

- **Classification**: 4-class softmax (Aadhaar, PAN, Fake, Other)
- **Authenticity**: Binary sigmoid output (0=fake, 1=authentic)

## Fake Detection Methods

1. **Color Analysis**: Checks for Aadhaar blue tint and detects plain white paper
2. **Border Tampering**: Edge detection to identify suspicious borders
3. **Handwritten Detection**: Identifies handwritten numbers on plain paper
4. **QR Validation**: Validates QR code presence and format
5. **Layout Analysis**: Detects document structure anomalies

## API Validation (Mock)

The system includes mock API validation functions:
- `mock_uidai_validation()`: Validates Aadhaar numbers
- `mock_it_validation()`: Validates PAN numbers

In production, replace these with actual API calls to UIDAI and IT Department services.

## Dataset Preparation

### Creating Synthetic Fake Documents

Use Albumentations for data augmentation to create synthetic fake documents:

```python
import albumentations as A

transform = A.Compose([
    A.RandomRotate90(p=0.3),
    A.Flip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    # ... more augmentations
])
```

### Recommended Dataset Size

- **Training**: ~700 images (distributed across classes)
- **Validation**: ~150 images
- **Test**: ~150 images
- **Total**: ~1K images

## Output Format

The system outputs JSON results in the following format:

```json
{
  "type": "Aadhaar",
  "type_confidence": 0.95,
  "authenticity": 0.92,
  "is_fake": false,
  "issues": [],
  "extracted_data": {
    "aadhaar_number": "123456789012",
    "pan_number": null,
    "text_length": 245
  },
  "api_validation": {
    "aadhaar": {
      "valid": true,
      "message": "Aadhaar format validated (mock)",
      "confidence": 0.85
    }
  }
}
```

## Troubleshooting

### Tesseract OCR Issues

If you encounter Tesseract errors:
- Ensure Tesseract is installed and in your PATH
- On macOS/Linux, verify with: `which tesseract`
- Set TESSDATA_PREFIX if needed: `export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata`

### Model Loading Issues

- Ensure model weights file exists at specified path
- Check that model architecture matches saved weights
- Verify TensorFlow/Keras versions are compatible

### Memory Issues

- Reduce batch size in training: `--batch_size 16`
- Use smaller image size (modify `target_size` in code)
- Enable mixed precision training

## Future Enhancements

- [ ] Real API integration for UIDAI and IT Department
- [ ] Support for more document types
- [ ] Improved fake detection algorithms
- [ ] Batch processing capabilities
- [ ] Model quantization for faster inference
- [ ] Docker containerization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational and research purposes. Ensure compliance with data privacy regulations when handling ID documents.

## Acknowledgments

- VGG16 architecture from Keras Applications
- Inspired by Magnum-Opus and documentClassification repositories
- Uses Tesseract OCR, OpenCV, and TensorFlow/Keras

## Contact

For questions or issues, please open an issue on the repository.

